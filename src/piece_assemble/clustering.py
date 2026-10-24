from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from multiprocessing import Manager, Pool
from typing import TYPE_CHECKING, Callable

import numpy as np
from skimage.transform import rescale
from tqdm import tqdm

from image import np_to_pil
from piece_assemble.cluster import Cluster
from piece_assemble.cycles import TransformationGraph
from piece_assemble.neighbors import ComplexityNeighborClassifier

if TYPE_CHECKING:
    from piece_assemble.cluster import ClusterScorer
    from piece_assemble.feature_extraction.base import FeatureExtractor
    from piece_assemble.matching.match import CompactMatch, Match
    from piece_assemble.piece import Piece


class Queue:
    def __init__(self, parallel: bool = False):
        self._queue = None
        self._list = []
        if parallel:
            self._queue = Manager().Queue()

    def empty(self) -> bool:
        if self._queue is not None:
            return self._queue.empty()
        else:
            return len(self._list) == 0

    def get(self) -> CompactMatch:
        if self._queue is not None:
            return self._queue.get()  # type: ignore
        else:
            return self._list.pop(0)  # type: ignore

    def put(self, item: CompactMatch | None) -> None:
        if self._queue is not None:
            self._queue.put(item)
        else:
            self._list.append(item)


class Clustering:
    """Class for assembling pieces into clusters."""

    def __init__(
        self,
        pieces: list[Piece],
        feature_extractor: FeatureExtractor,
        cluster_scorer: ClusterScorer,
    ) -> None:
        self.pieces = pieces
        self.feature_extractor = feature_extractor
        self.cluster_scorer = cluster_scorer
        self.all_ids = [piece.name for piece in pieces]

        # Optional parts of the algorithm, switchable for ablation purposes.
        # Overridden per run by __call__.
        self.use_icp = True
        self.use_trusted_clusters = True
        self.use_match_reuse = True
        self.check_consistency = False
        self.max_match_dist = None
        self.max_merge_dist = None
        self.max_merge_pair_dist = None
        self.finetune_dist_tol = None

        self.reset(True)
        self.set_logging(None)
        self.random = random.Random()

    @property
    def best_cluster(self) -> Cluster | None:
        if len(self.clusters) == 0:
            return None
        return self.clusters[0]  # type: ignore

    def reset(self, reset_all_matches: bool = False) -> None:
        self.clusters = []
        self.trusted_clusters = []
        self._i = 0
        self.all_pair_clusters = {}
        self.cluster_history = []
        self.assembled = False
        # Verified matches drained from the queue but not yet processed,
        # keyed by (id1, id2) pair and keeping only the best (lowest dist)
        # match seen so far for that pair. get_new_pair_clusters consumes
        # the globally-best-scoring entry each time, so matches are used in
        # quality order even though parallel workers push to the shared
        # queue in whatever order they each finish verifying -- and the
        # per-pair keying prevents the same pair's ~50 near-duplicate
        # candidates from monopolizing the front of that order.
        self._pending_matches: dict[tuple[str, str], CompactMatch] = {}
        self._pending_match_counter = 0
        self._reject_reasons: dict[str, int] = {}
        # One-shot guard for the redundant-match-trust cycle-closure check in
        # process_new_cluster: a given pair can only ever trigger it once.
        self._used_pair_clusters: set[frozenset[str]] = set()
        # Sort key (lower = better, same convention as CompactMatch.dist)
        # per _pending_matches key, computed from the match's real, fully-
        # verified cluster.score instead of the raw descriptor distance.
        self._pending_match_scores: dict[tuple, float] = {}

        if reset_all_matches:
            self.all_matches = None

    def _return_to_pending(self, match: CompactMatch) -> None:
        # Keep every verified candidate for a pair instead of collapsing to
        # just the single best-dist one: process_new_cluster's
        # all_pair_clusters re-sighting counter needs repeat sightings to
        # ever promote a borderline-complexity match. Keying on a counter
        # instead of the pair means every match gets its own slot, so the
        # pop-by-min-dist logic elsewhere needs no change.
        self._pending_match_counter += 1
        self._pending_matches[
            (match.id1, match.id2, self._pending_match_counter)
        ] = match

    def set_logging(
        self,
        output_images_path: str | None,
        store_new_matches: bool = True,
        store_old_matches: bool = False,
        store_trusted_clusters: bool = False,
    ) -> None:
        self._output_path = output_images_path
        self._store_new_matches = store_new_matches
        self._store_old_matches = store_old_matches
        self._store_trusted_clusters = store_trusted_clusters

        if output_images_path is None:
            return

        try:
            os.mkdir(output_images_path)
        except Exception:
            pass

    def find_candidate_matches(self, n_matches: int = 40000) -> None:
        # Pre-filter by is_initial_transform_valid() before applying the
        # n_used_matches budget, so the budget is spent on plausible
        # candidates rather than ones that can be rejected for free.
        self.all_matches = self.feature_extractor.find_all_matches_with_preprocessing(
            self.pieces
        )[:n_matches]

    def __call__(
        self,
        n_iters: int,
        trusted_cluster_config: dict,
        cluster_config: dict,
        icp_max_iters: int,
        icp_min_change: float,
        n_new_matches: int = 10,
        n_processes: int = 4,
        min_complexity: int = 2,
        n_used_matches: int = 40000,
        use_icp: bool = True,
        use_trusted_clusters: bool = True,
        use_match_reuse: bool = True,
        check_consistency: bool = False,
        max_match_dist: float | None = None,
        max_merge_dist: float | None = None,
        max_merge_pair_dist: float | None = None,
        finetune_dist_tol: float | None = None,
        seed: int | None = None,
    ) -> Cluster | None:
        # Building a Cluster needs a neighbor classifier, which a config read
        # from file doesn't carry. CompactMatch.to_cluster adds one, but only
        # once the first match is converted, which leaves anything constructing
        # a cluster before that (the cycle search, when running serially)
        # without it.
        # min_complexity=0: any nonzero border complexity counts as a
        # neighbor. The classifier's own default of 1 is stricter, which
        # lowers avg_neighbor_count (and therefore cluster.score, which
        # drives every selection/merge decision) for every cluster.
        cluster_config.setdefault(
            "neighbor_classifier",
            ComplexityNeighborClassifier(cluster_config["border_dist_tol"], min_complexity=0),
        )

        self.cluster_config = cluster_config
        self.use_icp = use_icp
        self.use_trusted_clusters = use_trusted_clusters
        self.use_match_reuse = use_match_reuse
        # Kept separate from cluster_config: that dict gets splatted as
        # **kwargs into Cluster(...) elsewhere (CompactMatch.to_cluster,
        # TransformationGraph.cycle_to_cluster), so adding match-verification
        # settings to it would break those constructor calls.
        self.check_consistency = check_consistency
        self.max_match_dist = max_match_dist
        self.max_merge_dist = max_merge_dist
        self.max_merge_pair_dist = max_merge_pair_dist
        self.finetune_dist_tol = finetune_dist_tol

        # The search draws on two independent sources of randomness: this
        # instance's own Random, and numpy's global one. Both have to be seeded
        # to make a run repeatable.
        if seed is not None:
            self.random.seed(seed)
            np.random.seed(seed)

        # ICP is applied in two places: refining each candidate pairwise match,
        # and refining piece placement after a cluster merge. Disabling it has
        # to cover both. icp() returns its initial transformation unchanged when
        # given no iterations, so zeroing the iteration counts is enough here;
        # the merge-time counts are zeroed at their own call sites.
        if not use_icp:
            icp_max_iters = 0

        if self.all_matches is None:
            self.find_candidate_matches(n_used_matches)

        # matches = self.all_matches.copy()

        queue = Queue(parallel=n_processes > 1)

        if n_processes > 1:
            self._run_parallel(
                n_iters,
                trusted_cluster_config,
                icp_max_iters,
                icp_min_change,
                n_new_matches,
                n_processes,
                min_complexity,
                queue,
            )
        else:
            self._run_serial(
                n_iters,
                trusted_cluster_config,
                icp_max_iters,
                icp_min_change,
                n_new_matches,
                min_complexity,
            )
        return self.best_cluster

    def _run_parallel(
        self,
        n_iters: int,
        trusted_cluster_config: dict,
        icp_max_iters: int,
        icp_min_change: float,
        n_new_matches: int,
        n_processes: int,
        min_complexity: int,
        queue: Queue,
    ) -> None:
        matches = self.all_matches.copy()  # type: ignore
        partitions = _partition_by_pair(matches, n_processes)
        with Pool(n_processes) as p:
            self._worker_count = n_processes
            for j in range(0, self._worker_count):
                p.apply_async(
                    matches_checker,
                    args=(
                        partitions[j],
                        queue,
                        self.cluster_config,
                        icp_max_iters,
                        icp_min_change,
                        self.check_consistency,
                        self.max_match_dist,
                    ),
                )

            self._workers_finished = 0
            # Block here until every worker has verified its entire share of
            # matches, instead of letting get_new_pair_clusters consume them
            # as they stream in from whichever worker happens to finish
            # first: with n_processes > 1 that streaming order is a race
            # condition, so the same seed could otherwise see a different
            # admission order from run to run. Once every match is in hand,
            # _pending_matches gives get_new_pair_clusters a fully sorted,
            # deterministic pop order for the rest of the run. The cost is
            # giving up the overlap between verification and the first
            # iterations' clustering work.
            while self._workers_finished < self._worker_count:
                item = queue.get()
                if item is None:
                    self._workers_finished += 1
                    print("Worker finished")
                    continue
                self._return_to_pending(item)

            # Rank by the real, fully-verified cluster.score instead of
            # match.dist (raw descriptor distance, only a proxy for
            # quality), now that every match is already in hand. Negated to
            # keep the existing "lower = better" convention _pending_matches'
            # pop sites already use for .dist.
            pieces_dict = {piece.name: piece for piece in self.pieces}
            for key, match in self._pending_matches.items():
                cluster = match.to_cluster(
                    self.cluster_scorer,
                    self.cluster_config,
                    pieces_dict=pieces_dict,
                )
                self._pending_match_scores[key] = -cluster.score

            for i in range(self._i, self._i + n_iters):
                if self.assembled:
                    break
                self._i = i + 1
                self.run_iteration(
                    i,
                    trusted_cluster_config,
                    n_new_matches,
                    min_complexity,
                )

    def _run_serial(
        self,
        n_iters: int,
        trusted_cluster_config: dict,
        icp_max_iters: int,
        icp_min_change: float,
        n_new_matches: int,
        min_complexity: int,
    ) -> None:
        all_compact_matches = []
        for match in tqdm(self.all_matches, desc="Generating matches"):
            if self.max_match_dist is not None and match.dist > self.max_match_dist:
                continue
            compact_match = match.verify(
                self.cluster_config["border_dist_tol"],
                icp_max_iters=icp_max_iters,
                icp_min_change=icp_min_change,
                check_consistency=self.check_consistency,
            )
            if compact_match is None:
                continue
            all_compact_matches.append(compact_match)
            self._return_to_pending(compact_match)

        cycles = self.get_cycles(all_compact_matches)
        self.update_trusted_clusters(
            cycles,
            lambda cluster: cluster_can_be_trusted(cluster, **trusted_cluster_config),
        )

        # Rank by the real, fully-verified cluster.score instead of
        # match.dist, same as _run_parallel -- see get_new_pair_clusters.
        pieces_dict = {piece.name: piece for piece in self.pieces}
        for key, pending_match in self._pending_matches.items():
            cluster = pending_match.to_cluster(
                self.cluster_scorer, self.cluster_config, pieces_dict=pieces_dict
            )
            self._pending_match_scores[key] = -cluster.score

        self._worker_count = 1
        self._workers_finished = 0

        for i in range(self._i, self._i + n_iters):
            if self.assembled:
                break
            self._i = i + 1
            self.run_iteration(
                i,
                trusted_cluster_config,
                n_new_matches,
                min_complexity,
            )

    def get_cycles(
        self,
        verified_matches: list[CompactMatch],
        angle_tol: float = 0.17,
        translation_tol: float = 30,
    ) -> list[Cluster]:
        # find_consistent_cycles's angle_tol/translation_tol gate which raw
        # cycles are even attempted: dropping this pre-filter entirely (i.e.
        # using find_cycles instead) was tried and reverted -- on a dense,
        # false-positive-heavy match graph the raw cycle count explodes, and
        # building a full Cluster (self-intersection, all-pairs border dist)
        # for every one of them is too expensive to do unconditionally. The
        # defaults mirror cluster.rotation_tol/translation_tol, but on
        # real-photo data individual pairwise ICP alignments carry enough
        # noise that composing 3-4 of them can exceed this tolerance even for
        # genuinely correct cycles, so a caller may need to loosen it.
        graph = TransformationGraph.from_matches(verified_matches)
        cycles = graph.find_consistent_cycles(
            3, angle_tol, translation_tol
        ) + graph.find_consistent_cycles(4, angle_tol, translation_tol)
        clusters = []
        pieces_dict = {piece.name: piece for piece in self.pieces}
        for cycle in cycles:
            cluster = graph.cycle_to_cluster(
                cycle, pieces_dict, self.cluster_scorer, self.cluster_config
            )
            if cluster is not None:
                clusters.append(cluster)
        return clusters

    def run_iteration(
        self,
        i: int,
        trusted_cluster_config: dict,
        n_new_matches: int,
        min_complexity: float,
    ) -> None:
        print(f"ITERATION {i}")

        new_pair_clusters = self.get_new_pair_clusters(
            n_new_matches, trusted_cluster_config, min_complexity
        )

        # A pair re-sighted at least 6 times across independently-verified
        # candidates is promoted to trusted unconditionally: by this point
        # repetition itself is the evidence, regardless of complexity.
        potential_trusted_clusters = []
        for all_found_pair_for_keys in self.all_pair_clusters.values():
            for pair_cluster in all_found_pair_for_keys:
                if pair_cluster["count"] >= 6:
                    potential_trusted_clusters.append(pair_cluster["cluster"])

        self.update_trusted_clusters(potential_trusted_clusters, lambda _: True)

        if self._store_trusted_clusters:
            self.store_iteration(f"{i}trusted", self.trusted_clusters)

        new_pair_clusters.sort(key=lambda cluster: cluster.score, reverse=True)

        if self._store_new_matches:
            self.store_iteration(f"{i}new_matches", new_pair_clusters)

        # Get interesting previous clusters
        previous_clusters = []
        use_previous_clusters = self.use_match_reuse and (
            len(self.clusters) >= 1
            and (
                len(self.clusters[0].piece_ids) > 0.5 * len(self.all_ids)
                or len(new_pair_clusters) < 5
            )
        )
        if use_previous_clusters:
            print("RECYCLING OLD MATCHES")
            previous_clusters = self.find_applicable_previous_clusters(
                self.clusters[0], max(5, n_new_matches - len(new_pair_clusters))
            )
            if len(previous_clusters) != 0 and self._store_old_matches:
                self.store_iteration(f"{i}old_matches", previous_clusters)
        self.clusters = self.use_new_matches(
            self.clusters + self.trusted_clusters, new_pair_clusters + previous_clusters
        )
        # Re-inject trusted clusters here too: use_new_matches already ran
        # cluster_selection (its per-piece best-score pick), which can drop a
        # trusted island that didn't literally win any piece this round --
        # e.g. a disjoint trusted chunk nothing has tried to merge with yet
        # simply won't beat whatever cluster currently holds those pieces on
        # score alone. Without this, such an island never gets a chance to be
        # a recombine() merge candidate again until something else happens to
        # touch it. cluster_selection runs again inside recombine() itself,
        # so including unneeded trusted clusters here is harmless -- they
        # just won't be selected if nothing better comes of them.
        self.clusters = self.recombine(self.clusters + self.trusted_clusters)
        self.clusters.sort(key=lambda cluster: cluster.score, reverse=True)

        self.store_iteration(f"{i}iter", self.clusters)
        self.cluster_history.append(self.clusters)

        if (
            self._worker_count == self._workers_finished
            and len(previous_clusters) == 0
            and use_previous_clusters
        ):
            self.assembled = True

        if self.best_cluster is None:
            return

        if len(self.clusters) > 0 and len(self.best_cluster.piece_ids) == len(
            self.all_ids
        ):
            self.assembled = True

    def get_new_pair_clusters(
        self,
        n_new_matches: int,
        trusted_cluster_config: dict,
        min_complexity: float,
    ) -> list[Cluster]:
        # Pop candidates from _pending_matches in real-cluster.score order
        # (fully populated and scored up front by _run_parallel's pre-drain,
        # or incrementally by _run_serial) -- deterministic rather than
        # whatever order parallel workers happen to finish verifying in.
        new_pair_clusters = []
        self._reject_reasons = {}
        pieces_dict = {piece.name: piece for piece in self.pieces}

        with tqdm(desc="Generating new matches", total=n_new_matches) as pbar:
            while len(new_pair_clusters) < n_new_matches:
                if not self._pending_matches:
                    break
                best_key = min(
                    self._pending_matches,
                    key=lambda k: self._pending_match_scores.get(
                        k, self._pending_matches[k].dist
                    ),
                )
                new_match = self._pending_matches.pop(best_key)

                new_cluster = new_match.to_cluster(
                    self.cluster_scorer, self.cluster_config, pieces_dict=pieces_dict
                )
                new_cluster = self.process_new_cluster(
                    new_cluster, trusted_cluster_config, min_complexity
                )
                if new_cluster is not None:
                    new_pair_clusters.append(new_cluster)
                    pbar.update(1)
            return new_pair_clusters


    def process_new_cluster(
        self, new_cluster: Cluster, trusted_cluster_config: dict, min_complexity: float
    ) -> Cluster | None:
        # Redundant-match-trust cycle-closure check: finds cycles of
        # ARBITRARY length -- if a freshly-verified 2-piece match's both
        # pieces are already present somewhere in an existing regular
        # cluster, and this new match's own placement agrees with that
        # cluster's, promote the match itself to trusted (lambda _: True,
        # genuinely unconditional -- there's no threshold here to inflate by
        # repetition, so no quality gate is needed on the promotion either).
        # _used_pair_clusters is a one-shot guard: a given pair can only
        # ever trigger this once.
        new_keys = frozenset(new_cluster.piece_ids)
        if new_keys not in self._used_pair_clusters:
            self._used_pair_clusters.add(new_keys)
            for existing in self.clusters:
                if existing is new_cluster or not new_keys.issubset(
                    existing.piece_ids
                ):
                    continue
                if not existing.common_pieces_match(new_cluster):
                    continue
                promoted = self.update_trusted_clusters(
                    [new_cluster], lambda _: True
                )
                new_cluster = promoted[0] if promoted else None
                if new_cluster is None:
                    break

        if new_cluster is None:
            return None

        if type(min_complexity) in (int, float):
            min_complexity = [min_complexity, min_complexity]
        if new_cluster.complexity < min_complexity[1]:
            self._reject_reasons["complexity_floor"] = (
                self._reject_reasons.get("complexity_floor", 0) + 1
            )
            return None

        before_trust_count = len(self.trusted_clusters)
        new_cluster_list = self.update_trusted_clusters(
            [new_cluster],
            lambda cluster: cluster_can_be_trusted(cluster, **trusted_cluster_config),
        )
        absorbed_into_trust = (
            len(new_cluster_list) == 0 and len(self.trusted_clusters) > before_trust_count
        )
        dropped_by_trust_merge = len(new_cluster_list) == 0 and not absorbed_into_trust

        before_check_count = len(new_cluster_list)
        new_cluster_list = self._check_new_clusters(new_cluster_list)
        rejected_by_check = before_check_count > 0 and len(new_cluster_list) == 0

        if len(new_cluster_list) == 0:
            if absorbed_into_trust:
                self._reject_reasons["absorbed_into_trust"] = (
                    self._reject_reasons.get("absorbed_into_trust", 0) + 1
                )
            elif dropped_by_trust_merge:
                self._reject_reasons["dropped_by_trust_merge"] = (
                    self._reject_reasons.get("dropped_by_trust_merge", 0) + 1
                )
            elif rejected_by_check:
                self._reject_reasons["rejected_by_check_new_clusters"] = (
                    self._reject_reasons.get("rejected_by_check_new_clusters", 0) + 1
                )
            return None
        new_cluster = new_cluster_list[0]

        new_keys = frozenset(new_cluster.piece_ids)
        is_duplicate = False
        all_found_pair_for_keys = self.all_pair_clusters.get(new_keys, [])
        for pair_cluster_dict in all_found_pair_for_keys:
            pair_cluster = pair_cluster_dict["cluster"]
            if pair_cluster.can_be_merged(new_cluster):
                if new_cluster.score > pair_cluster.score:
                    pair_cluster_dict["cluster"] = new_cluster
                pair_cluster_dict["count"] = pair_cluster_dict["count"] + 1
                is_duplicate = True
                break

        if is_duplicate:
            return None

        all_found_pair_for_keys.append({"cluster": new_cluster, "count": 1})
        self.all_pair_clusters[new_keys] = all_found_pair_for_keys

        if new_cluster.complexity >= min_complexity[0]:
            # Use this match now only if it's interesting enough, otherwise
            # just remember it to use it later
            return new_cluster

        return None

    def combine(
        self,
        cluster1: Cluster,
        cluster2: Cluster,
        max_cluster_size: int | None,
        finetune_iters: int | None = None,
    ) -> Cluster | None:
        """Combine two clusters, if possible.

        Parameters
        ----------
        cluster1
            First cluster -- the merge's anchor frame (find_unifying_
            transform resolves a conflicting shared-piece placement in
            cluster1's favor). Callers that want a randomized anchor should
            pick which cluster is cluster1 before calling.
        cluster2
            Second cluster.
        max_cluster_size
            Cluster won't be merged if their total number of pieces exceeds this limit.

        Returns
        -------
        new_cluster
            New combined cluster.
        """
        if (
            max_cluster_size is not None
            and len(cluster1.piece_ids.union(cluster2.piece_ids)) > max_cluster_size
        ):
            return None
        if cluster1.piece_ids.isdisjoint(cluster2.piece_ids):
            return None

        if cluster1.piece_ids.issubset(
            cluster2.piece_ids
        ) or cluster2.piece_ids.issubset(cluster1.piece_ids):
            return None

        if finetune_iters is None:
            finetune_iters = 20
        if not self.use_icp:
            finetune_iters = 0

        try:
            new_cluster = cluster1.merge(
                cluster2,
                finetune_iters=finetune_iters,
                finetune_dist_tol=self.finetune_dist_tol,
            )
        except Exception:
            return None

        # Merging never fails purely because the result is geometrically
        # inconsistent: `merge` only raises for a disagreeing anchor-piece
        # transform or excessive self-intersection (whose tolerance actually
        # *loosens* as clusters grow, via self_intersection_tol * (log2(n)+1)
        # in Cluster.merge). cluster.dist -- the same all-pairs border-gap
        # residual already used to judge ordinary 2-piece matches and to
        # decide trust -- is otherwise only used for scoring, never to reject
        # a merge outright. That means a badly-aligned merge of two large,
        # internally-drifted clusters can still go through as long as no
        # exception fires, and once it does, nothing later undoes it: if it's
        # the only candidate covering those piece IDs that round,
        # cluster_selection has no better alternative to prefer instead. This
        # gate makes that failure mode explicit and preventable by rejecting
        # the merge outright, the same way Match.verify already rejects a
        # 2-piece candidate whose residual is too high.
        if self.max_merge_dist is not None and new_cluster.dist > self.max_merge_dist:
            return None
        # max_merge_dist uses the cluster-wide mean, which a single
        # badly-placed piece can't move much once the cluster has enough
        # other (correct) pairs -- confirmed on owl_101pcs: a merge that
        # replaced 10 correct pieces with 10 wrong ones left cluster.dist
        # nearly unchanged (0.995 -> 0.973), so it sailed through this gate.
        # max_pair_dist takes the max over per-pair means instead of
        # pooling every point from every pair together, so one bad
        # piece-to-piece fit stays visible regardless of cluster size.
        if (
            self.max_merge_pair_dist is not None
            and new_cluster.max_pair_dist > self.max_merge_pair_dist
        ):
            return None
        return new_cluster

    def use_new_matches(
        self,
        clusters: list[Cluster],
        pair_clusters: list[Cluster],
        max_cluster_size: int | None = None,
    ) -> list[Cluster]:
        clusters = self.apply_trusted_clusters(clusters)
        clusters = self.cluster_selection(clusters)
        pair_clusters.sort(key=lambda cluster: cluster.score, reverse=True)
        for c1 in tqdm(pair_clusters, desc="Using new matches"):
            clusters_dict = {
                frozenset(cluster.piece_ids): cluster for cluster in clusters + [c1]
            }

            for c2 in clusters:
                new_cluster = self.combine(c1, c2, max_cluster_size, finetune_iters=10)
                if new_cluster is None:
                    continue
                key = frozenset(new_cluster.piece_ids)
                if key in clusters_dict.keys():
                    if clusters_dict[key].score >= new_cluster.score:
                        continue

                clusters_dict[key] = new_cluster

            clusters = list(clusters_dict.values())
            clusters = self.apply_trusted_clusters(clusters)
            clusters = self.cluster_selection(clusters)
        return clusters

    def recombine(
        self, clusters: list[Cluster], max_cluster_size: int | None = None
    ) -> list[Cluster]:
        """Create new clusters from given clusters via merging.

        Merges together all clusters which can be merged.
        Runs in iterations.

        Parameters
        ----------
        clusters
            List of clusters.
        max_cluster_size
            Cluster won't be merged if their total number of pieces exceeds this limit.

        Returns
        -------
        recombined_clusters
            List of clusters created from the original clusters with merging.
        """
        new_clusters_added = True
        while new_clusters_added:
            new_clusters_added = False
            new_clusters_dict = {
                frozenset(cluster.piece_ids): cluster for cluster in clusters
            }
            # Tried sorting by score descending instead of shuffling (so
            # higher-scoring clusters get first crack at finding a partner
            # each pass, since a cluster is marked "used" after one success
            # and can't try again until the next pass) -- confirmed worse in
            # practice (ratio 0.881 -> 0.495 on owl_101pcs): always giving
            # the same few big clusters first priority starves smaller ones
            # of a chance to find *their* best partner before being crowded
            # out, pass after pass. The random shuffle's exploration value
            # outweighs the theoretical benefit of greedy-by-score ordering.
            self.random.shuffle(clusters)

            used_pieces = set()
            with tqdm(desc="Recombining", total=math.comb(len(clusters), 2)) as pbar:
                for i, c1 in enumerate(clusters[:-1]):
                    key1 = frozenset(c1.piece_ids)
                    if key1 in used_pieces:
                        pbar.update(len(clusters[i + 1 :]))
                        continue
                    for c2 in clusters[i + 1 :]:
                        pbar.update(1)
                        key2 = frozenset(c2.piece_ids)
                        if key1 in used_pieces or key2 in used_pieces:
                            continue

                        # Randomize which cluster anchors the merge frame
                        # (combine()'s cluster1): find_unifying_transform
                        # resolves a conflicting shared-piece placement in
                        # the anchor's favor, and which side wins can change
                        # the result on a true ambiguity. Only affects this
                        # call, not c1/c2 themselves (used below by key1/
                        # key2/used_pieces).
                        merge_c1, merge_c2 = (
                            (c2, c1) if self.random.random() < 0.5 else (c1, c2)
                        )
                        new_cluster = self.combine(merge_c1, merge_c2, max_cluster_size)
                        if new_cluster is None:
                            continue

                        new_key = frozenset(new_cluster.piece_ids)
                        # Dominated-cluster pruning: dropping a merge input
                        # once it's been absorbed into a larger result avoids
                        # repeatedly re-attempting merges against clusters
                        # that can no longer win (a large speedup on owl_101pcs).
                        dominated_keys = {
                            key for key in (key1, key2) if key != new_key
                        }
                        if new_key in new_clusters_dict.keys():
                            if new_clusters_dict[new_key].score < new_cluster.score:
                                new_clusters_dict[new_key] = new_cluster
                                new_clusters_added = True
                        else:
                            new_clusters_dict[new_key] = new_cluster
                            new_clusters_added = True
                        for key in dominated_keys:
                            if len(key) > 2 and key in new_clusters_dict:
                                del new_clusters_dict[key]
                                new_clusters_added = True
                        used_pieces.update(
                            {key for key in (key1, key2) if len(key) > 2}
                        )

            if len(new_clusters_dict.values()) == len(clusters):
                new_clusters_added = False

            prev_cluster_len = len(clusters)
            clusters = list(new_clusters_dict.values())
            clusters = self.cluster_selection(clusters)
            if prev_cluster_len == len(clusters):
                new_clusters_added = False

        clusters = self.apply_trusted_clusters(clusters)
        if len(clusters) == 0:
            return self.trusted_clusters
        clusters.sort(key=lambda cluster: cluster.score, reverse=True)
        return self._prune_dominated(clusters)

    @staticmethod
    def _prune_dominated(clusters: list[Cluster]) -> list[Cluster]:
        """Drop any cluster that is a strict subset of an earlier
        (assumed higher-scoring) one in the same list. Needed both after
        cluster_selection's own voting and again here after
        apply_trusted_clusters, since applying a trusted cluster can itself
        create a new domination relationship cluster_selection never had a
        chance to see."""
        pruned: list[Cluster] = []
        for cluster in clusters:
            if any(cluster.piece_ids < kept.piece_ids for kept in pruned):
                continue
            pruned.append(cluster)
        return pruned

    def cluster_selection(self, clusters: list[Cluster]) -> list[Cluster]:
        """Select best cluster representative.

        Parameters
        ----------
        clusters
            List of clusters

        Returns
        -------
        selected_clusters
            List of selected clusters.
        """
        clusters = clusters + self.trusted_clusters
        clusters.sort(key=lambda cluster: cluster.score, reverse=True)
        clusters_by_piece = {key: [] for key in self.all_ids}

        for i, cluster in enumerate(clusters):
            for piece_id in cluster.piece_ids:
                clusters_by_piece[piece_id].append(i)

        selected_cluster_idxs = set()

        for piece_id, cluster_list in clusters_by_piece.items():
            if len(cluster_list) == 0:
                continue
            new_i = cluster_list.pop(0)

            if new_i is not None:
                selected_cluster_idxs.add(new_i)

        selection = [clusters[i] for i in selected_cluster_idxs]
        selection.sort(key=lambda cluster: cluster.score, reverse=True)

        # combine() refuses to even attempt a merge between a cluster and
        # its own subset/superset, so recombine()'s own dominated-cluster
        # pruning never gets a chance to run for this exact relationship --
        # prune it here instead.
        pruned_selection = []
        for cluster in selection:
            if any(
                cluster.piece_ids < kept.piece_ids for kept in pruned_selection
            ):
                continue
            pruned_selection.append(cluster)
        return pruned_selection

    def apply_trusted_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        """Extend given clusters by trusted clusters.

        Parameters
        ----------
        clusters

        Returns
        -------
        new_cluster
            Extended clusters (only valid ones).
        """
        if len(self.trusted_clusters) == 0:
            return clusters
        new_clusters = {}
        for cluster in tqdm(clusters, "Applying trusted clusters"):
            for trusted_cluster in self.trusted_clusters:
                if trusted_cluster.piece_ids.issubset(
                    cluster.piece_ids
                ) and trusted_cluster.common_pieces_match(cluster):
                    continue
                if cluster.piece_ids.isdisjoint(trusted_cluster.piece_ids):
                    continue
                try:
                    cluster = trusted_cluster.merge(
                        cluster,
                        try_fix=True,
                        finetune_iters=20 if self.use_icp else 0,
                        finetune_dist_tol=self.finetune_dist_tol,
                    )
                except Exception:
                    cluster = None
                    break

            if cluster is None:
                continue

            key = frozenset(cluster.piece_ids)
            if key in new_clusters.keys() and new_clusters[key].score >= cluster.score:
                continue
            new_clusters[key] = cluster

        return list(new_clusters.values())

    def update_trusted_clusters(
        self, clusters: list[Cluster], trust_function: Callable
    ) -> list[Cluster]:
        # With trusted clusters disabled, self.trusted_clusters stays empty and
        # every cluster is passed on as an ordinary one. All the other places
        # that consume self.trusted_clusters degrade to no-ops on an empty list.
        if not self.use_trusted_clusters:
            return clusters

        other_clusters = []
        for cluster in clusters:
            used_trusted_clusters = []
            for trusted_cluster in self.trusted_clusters:
                if cluster.piece_ids.issubset(trusted_cluster.piece_ids):
                    cluster = None
                    break
                if cluster.piece_ids.isdisjoint(trusted_cluster.piece_ids):
                    continue
                if trust_function(cluster):
                    try:
                        cluster = cluster.merge(
                            trusted_cluster,
                            try_fix=False,
                            finetune_iters=20 if self.use_icp else 0,
                            finetune_dist_tol=self.finetune_dist_tol,
                        )
                        used_trusted_clusters.append(trusted_cluster)
                    except Exception:
                        cluster = None
                        break

            if cluster is None:
                continue

            if trust_function(cluster) or len(used_trusted_clusters) > 0:
                self.trusted_clusters.append(cluster)
                for trusted_cluster in used_trusted_clusters:
                    self.trusted_clusters.remove(trusted_cluster)

            else:
                other_clusters.append(cluster)

        return other_clusters

    def _check_new_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        selected_clusters = []
        for cluster in clusters:
            can_be_used = True
            for trusted_cluster in self.trusted_clusters:
                if cluster.piece_ids.isdisjoint(trusted_cluster.piece_ids):
                    continue
                if not cluster.can_be_merged(trusted_cluster):
                    can_be_used = False
                    break
            if can_be_used:
                selected_clusters.append(cluster)
        return selected_clusters

    def find_applicable_previous_clusters(
        self, cluster: Cluster, max_count: int
    ) -> list[Cluster]:
        """
        Finds the suitable previously find pair clusters for the given cluster.

        Parameters
        ----------
        best_cluster
            The best cluster.
        max_count
            The maximum number of clusters to return.

        Returns
        -------
        list
            A list of applicable previous clusters.
        """

        previous_cluster_list = []
        cluster_pairs = cluster.get_neighbor_pairs()
        for keys, pair_previous_clusters in self.all_pair_clusters.items():
            if keys not in cluster_pairs:
                pair_previous_clusters = [
                    cluster["cluster"] for cluster in pair_previous_clusters
                ]
                pair_previous_clusters.sort(
                    key=lambda cluster: cluster.score, reverse=True
                )
                if keys.issubset(cluster.piece_ids) and pair_previous_clusters[
                    0
                ].common_pieces_match(cluster):
                    continue
                previous_cluster_list.append(pair_previous_clusters[0])

        previous_cluster_list = self._check_new_clusters(previous_cluster_list)
        if len(previous_cluster_list) > 0:
            worst_score = min([cluster.score for cluster in previous_cluster_list])
            # Candidates that only share a single piece with the current
            # cluster (a bridge into otherwise-unconnected territory) get
            # their score margin raised to the 4th power, so a strong bridge
            # candidate dominates the draw instead of being drowned out by
            # many weaker same-cluster candidates with a smaller individual
            # margin each.
            probabilities = np.array(
                [
                    (
                        (c.score - worst_score) ** 4
                        if len(c.piece_ids.intersection(cluster.piece_ids)) == 1
                        else (c.score - worst_score)
                    )
                    for c in previous_cluster_list
                ]
            )
            probabilities = probabilities**0.5 + 0.0000001
            probabilities = probabilities / sum(probabilities)

            previous_clusters = list(
                np.random.choice(
                    previous_cluster_list,
                    min(max_count, len(previous_cluster_list)),
                    replace=False,
                    p=probabilities,
                )
            )
            return previous_clusters
        return []

    def store_iteration(self, name: str, clusters: list[Cluster]) -> None:
        """
        Store the given clusters in the specified directory.

        Parameters
        ----------
        name
            The name of the directory.
        clusters
            The list of clusters to be stored.

        """
        if self._output_path is None:
            return

        base_dir = f"{self._output_path}/{name}"

        try:
            os.mkdir(base_dir)
        except Exception:
            shutil.rmtree(base_dir)
            os.mkdir(base_dir)

        def get_image_path(i: int, cluster: Cluster) -> str:
            image_name = (
                f"{name}_{i:03d}_score{cluster.score:.2f}"
                + f"_color{cluster.color_dist:.3f}_dist{cluster.dist:.3f}"
                + f"_complexity{cluster.complexity:.3f}.png"
            )
            return os.path.join(base_dir, image_name)

        for i, cluster in enumerate(clusters):
            img = np_to_pil(rescale(cluster.draw(), 0.5, channel_axis=2))
            image_path = get_image_path(i, cluster)
            img.save(image_path)
            # Sidecar with the same basename as the image, in the same
            # {transformed_pieces, neighbors} schema as a puzzle's own
            # ground-truth pieces.json (Transformation.to_dict() matches
            # Transformation.from_dict()'s expected format exactly), so a
            # stored cluster can be loaded and compared against ground truth
            # with the same code that already reads pieces.json, instead of
            # guessing piece placement from colors/shapes by eye.
            with open(os.path.splitext(image_path)[0] + ".json", "w") as f:
                json.dump(cluster.to_dict(), f)


def cluster_can_be_trusted(
    cluster: Cluster,
    complexity_threshold: float,
    dist_threshold: float,
    color_threshold: float,
) -> bool:
    """
    Check if a cluster can be trusted.

    Parameters
    ----------
    cluster
        The cluster to be checked.
    complexity_threshold
        The threshold for the cluster's complexity.
    dist_threshold
        The threshold for the cluster's distance.
    color_threshold
        The threshold for the cluster's color distance.

    Returns
    -------
    bool
        True if the cluster can be trusted, False otherwise.

    """
    return (  # type: ignore
        cluster.complexity > complexity_threshold * (len(cluster.piece_ids) - 1)
        and cluster.dist < dist_threshold
        and cluster.color_dist < color_threshold
    )


def _unordered_pair(id1: str, id2: str) -> tuple[str, str]:
    return (id1, id2) if id1 <= id2 else (id2, id1)


def _partition_by_pair(matches: list[Match], n_processes: int) -> list[list[Match]]:
    """Split matches across workers by which piece-pair they belong to,
    keeping every candidate for a given pair in the same partition.

    Splitting by plain list position (`matches[j::n_processes]`) scatters a
    single pair's ~50 near-duplicate candidates across every worker, since
    striding by index cycles through all workers regardless of which pair a
    match belongs to. Hash-grouping instead lets one worker verify a pair's
    near-duplicates back-to-back, so that pair's true best candidate
    stabilizes in `_pending_matches` quickly; round-robin striding scatters
    them across every worker instead, which can delay that stabilization
    since it now depends on all workers reaching their scattered share --
    confirmed to measurably regress `synth_color_50pcs`'s reliability, so
    hash-grouping is used instead. Keeps each partition's relative
    distance-sorted order, same as plain striding did.
    """
    partitions: list[list[Match]] = [[] for _ in range(n_processes)]
    for match in matches:
        pair = _unordered_pair(match.id1, match.id2)
        partitions[hash(pair) % n_processes].append(match)
    return partitions


def matches_checker(
    matches: list[Match],
    queue: Queue,
    config: dict,
    icp_max_iters: int,
    icp_min_change: float = 0.5,
    check_consistency: bool = False,
    max_match_dist: float | None = None,
) -> None:
    try:
        for match in matches:
            if max_match_dist is not None and match.dist > max_match_dist:
                continue
            # A single geometrically-degenerate candidate (e.g. shapely
            # raising on a self-intersecting polygon from a rough initial
            # transform) must not kill the rest of this worker's share of
            # matches -- previously an unguarded exception here propagated
            # out of the loop entirely, silently dropping every remaining
            # match assigned to this worker for the rest of the run.
            try:
                compact_match = match.verify(
                    config["border_dist_tol"],
                    icp_max_iters=icp_max_iters,
                    icp_min_change=icp_min_change,
                    check_consistency=check_consistency,
                )
            except Exception as e:
                print(f"Match verification failed for {match}: {e}")
                continue
            if compact_match is None:
                continue
            queue.put(compact_match)
    except Exception as e:
        print(e)
    finally:
        queue.put(None)
