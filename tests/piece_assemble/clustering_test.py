from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from piece_assemble.clustering import Clustering


def make_clustering() -> Clustering:
    """A Clustering instance with no real pieces, for exercising the parts of the
    algorithm that don't depend on actual piece geometry."""
    return Clustering([], MagicMock(), MagicMock())


@pytest.mark.parametrize("use_trusted_clusters", [True, False])
def test_update_trusted_clusters_can_be_disabled(use_trusted_clusters: bool) -> None:
    clustering = make_clustering()
    clustering.use_trusted_clusters = use_trusted_clusters

    cluster = MagicMock()
    cluster.piece_ids = {"0", "1"}

    remaining = clustering.update_trusted_clusters([cluster], lambda _: True)

    if use_trusted_clusters:
        # The cluster is trusted, so it is absorbed rather than passed on.
        assert clustering.trusted_clusters == [cluster]
        assert remaining == []
    else:
        # Nothing is ever trusted, and every cluster stays an ordinary one.
        assert clustering.trusted_clusters == []
        assert remaining == [cluster]


@pytest.mark.parametrize("use_icp", [True, False])
def test_combine_skips_finetuning_when_icp_disabled(use_icp: bool) -> None:
    clustering = make_clustering()
    clustering.use_icp = use_icp

    # Clusters are only merged if they overlap without either containing the
    # other, so give them one piece in common.
    cluster1 = MagicMock()
    cluster1.piece_ids = {"0", "1"}
    cluster2 = MagicMock()
    cluster2.piece_ids = {"1", "2"}

    clustering.combine(cluster1, cluster2, max_cluster_size=None, randomize_order=False)

    _, kwargs = cluster1.merge.call_args
    assert kwargs["finetune_iters"] == (5 if use_icp else 0)


@pytest.mark.parametrize("use_icp", [True, False])
def test_call_zeroes_pairwise_icp_iterations_when_disabled(use_icp: bool) -> None:
    clustering = make_clustering()
    clustering.all_matches = []

    with patch.object(Clustering, "_run_serial") as run_serial:
        clustering(
            n_iters=1,
            trusted_cluster_config={},
            cluster_config={},
            icp_max_iters=100,
            icp_min_change=0.3,
            n_processes=1,
            use_icp=use_icp,
        )

    # icp_max_iters is the third positional argument of _run_serial.
    args, _ = run_serial.call_args
    assert args[2] == (100 if use_icp else 0)
    assert clustering.use_icp is use_icp


def run_once_with_seed(seed: int | None) -> tuple[float, float]:
    """Draw from both of the search's random sources after seeding with `seed`."""
    clustering = make_clustering()
    clustering.all_matches = []

    with patch.object(Clustering, "_run_serial"):
        clustering(
            n_iters=1,
            trusted_cluster_config={},
            cluster_config={},
            icp_max_iters=10,
            icp_min_change=0.3,
            n_processes=1,
            seed=seed,
        )

    return clustering.random.random(), float(np.random.rand())


def test_seed_makes_both_random_sources_repeatable() -> None:
    assert run_once_with_seed(1234) == run_once_with_seed(1234)


def test_different_seeds_give_different_draws() -> None:
    assert run_once_with_seed(1234) != run_once_with_seed(5678)


@pytest.mark.parametrize("use_match_reuse", [True, False])
def test_match_reuse_can_be_disabled(use_match_reuse: bool) -> None:
    clustering = make_clustering()
    clustering.use_match_reuse = use_match_reuse
    clustering.all_ids = ["0", "1"]
    # Normally set up by _run_serial/_run_parallel, which this test bypasses.
    clustering._worker_count = 1
    clustering._workers_finished = 0

    # A single cluster already covering every piece is exactly the situation
    # that would otherwise trigger recycling of previously unused matches.
    big_cluster = MagicMock()
    big_cluster.piece_ids = {"0", "1"}
    big_cluster.score = 1.0
    clustering.clusters = [big_cluster]

    with (
        patch.object(Clustering, "get_new_pair_clusters", return_value=[]),
        patch.object(
            Clustering, "find_applicable_previous_clusters", return_value=[]
        ) as find_previous,
        patch.object(Clustering, "use_new_matches", return_value=[]),
        patch.object(Clustering, "recombine", return_value=[]),
    ):
        clustering.run_iteration(
            i=0,
            trusted_cluster_config={},
            n_new_matches=10,
            min_complexity=2,
            queue=MagicMock(),
        )

    assert find_previous.called is use_match_reuse
