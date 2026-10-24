"""
Script for evaluating the assembly algorithm against ground truth puzzles.

Reports two groups of metrics per puzzle: local-stage metrics score the
pairwise matching stage alone (does it find the correct adjacent piece
pairs, before any global assembly happens?), and global-stage metrics score
the final assembly the search settles on. See LOCAL_METRIC_NAMES /
GLOBAL_METRIC_NAMES below for the exact fields.

Each puzzle can be assembled several times with different random seeds, since
the assembly search (the global stage) is stochastic and a single run is not
representative. The local-stage metrics involve no randomness and do not
change across repeats of the same puzzle and config.

usage: eval_assembly.py [-h] [--repeats REPEATS] [--seed SEED]
                        [--output-dir OUTPUT_DIR] [--label LABEL]
                        config puzzles_path dataset_path

positional arguments:
  config        Configuration file, same structure as used for run_assembly.py.
                Only img_path gets overridden, once per evaluated puzzle.
  puzzles_path  Path to a file listing the puzzle directories (one per line,
                relative to dataset_path) that should be used for evaluation.
  dataset_path  Path to the directory containing the puzzle directories listed
                in puzzles_path. Each puzzle directory must contain piece
                images/masks plus a pieces.json ground truth, as produced by
                puzzle_generator/generate.py.

Reproducibility
---------------
Repeating a run exactly takes more than --seed. All three of these are needed:

  --seed N            seeds the search's own randomness
  n_processes: 1      several workers return verified matches in whatever order
                      they finish, and that order changes the outcome
  PYTHONHASHSEED=0    python randomises string hashing per process, which
                      changes the iteration order of the sets of piece ids the
                      algorithm walks over

Serial runs are noticeably slower, so this is worth doing for results that have
to be reproducible, and not for large sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from copy import deepcopy

from piece_assemble.config import load_config
from piece_assemble.evaluation import (
    cluster_to_transformations,
    evaluate,
    evaluate_matching,
)
from piece_assemble.load import load_puzzle
from piece_assemble.tools.run_assembly import build_pieces, run_prediction

# "Local"-stage metrics score the pairwise matching stage alone (does it find
# the right adjacent piece pairs?), independent of what the global assembly
# stage goes on to do with that output. "Global"-stage metrics score the
# final assembly (largest_cluster_ratio, num_of_clusters) and the placement
# accuracy of pieces the global stage did assemble correctly
# (angle/translation/transformation_error).
LOCAL_METRIC_NAMES = [
    "match_precision",
    "match_recall",
    "match_f1",
    "match_tp",
    "match_fp",
    "match_fn",
]
GLOBAL_METRIC_NAMES = [
    "angle_error",
    "translation_error",
    "transformation_error",
    "largest_cluster_ratio",
    "num_of_clusters",
    "runtime_seconds",
]
METRIC_NAMES = LOCAL_METRIC_NAMES + GLOBAL_METRIC_NAMES


def evaluate_puzzle(config: dict, puzzle_dir: str, seed: int | None = None) -> dict:
    """Assemble a single puzzle once and score both the pairwise matching
    stage and the final assembly against ground truth.

    Parameters
    ----------
    config
        The assembly configuration. `img_path` is overridden with `puzzle_dir`.
    puzzle_dir
        Directory holding the puzzle's pieces and its pieces.json ground truth.
    seed
        Seed for the assembly search, for repeatable runs. The local matching
        metrics do not depend on it: pairwise match generation and
        verification involve no randomness, so they agree across repeats of
        the same puzzle and config.

    Returns
    -------
    The local-stage metrics from `evaluation.evaluate_matching`, the
    global-stage metrics from `evaluation.evaluate`, and the wall-clock
    runtime of the assembly search.
    """
    gt_pieces, neighbors = load_puzzle(puzzle_dir, scale=config["piece"]["scale"])
    gt_transformations = {
        piece.name: piece.transformation for piece in gt_pieces.values()
    }

    puzzle_config = deepcopy(config)
    puzzle_config["img_path"] = puzzle_dir
    if seed is not None:
        puzzle_config["clustering"]["seed"] = seed
    output_images_path = config["logging"].get("output_images_path")
    if output_images_path:
        puzzle_config["logging"]["output_images_path"] = os.path.join(
            output_images_path, os.path.basename(puzzle_dir)
        )

    # use_icp also gates pairwise match verification (not just the global
    # merge-time refinement), so the local metric has to respect it the same
    # way Clustering.__call__ does, or it would silently measure a different
    # ICP setting than the assembly run right below it.
    use_icp = puzzle_config["clustering"].get("use_icp", True)
    icp_max_iters = puzzle_config["clustering"]["icp_max_iters"] if use_icp else 0

    pieces, feature_extractor = build_pieces(puzzle_config)
    metrics = evaluate_matching(
        pieces,
        feature_extractor,
        neighbors,
        puzzle_config["cluster"]["border_dist_tol"],
        icp_max_iters,
        puzzle_config["clustering"]["icp_min_change"],
        puzzle_config["clustering"]["n_used_matches"],
        check_consistency=puzzle_config["clustering"].get("check_consistency", False),
        max_match_dist=puzzle_config["clustering"].get("max_match_dist", None),
    )

    start = time.monotonic()
    clusters = run_prediction(puzzle_config)
    runtime = time.monotonic() - start

    metrics.update(
        evaluate(
            [cluster_to_transformations(cluster) for cluster in clusters],
            gt_transformations,
            neighbors,
            config["cluster"]["rotation_tol"],
            config["cluster"]["translation_tol"],
        )
    )
    metrics["runtime_seconds"] = runtime
    return dict(metrics)


def aggregate(runs: list[dict]) -> dict:
    """Mean and standard deviation of every metric over a set of runs.

    Standard deviation is reported as 0 for a single run, which has none to
    speak of, rather than being left undefined.
    """
    aggregated = {}
    for name in METRIC_NAMES:
        values = [run[name] for run in runs if name in run]
        if not values:
            continue
        aggregated[f"{name}_mean"] = statistics.fmean(values)
        aggregated[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return aggregated


def write_results(output_dir: str, label: str, runs: list[dict]) -> tuple[str, str]:
    """Write the individual runs as CSV and their aggregate as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{label}_runs.csv")
    fields = ["puzzle", "seed"] + METRIC_NAMES
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)

    json_path = os.path.join(output_dir, f"{label}_summary.json")
    per_puzzle = sorted({run["puzzle"] for run in runs})
    summary = {
        "label": label,
        "num_puzzles": len(per_puzzle),
        "num_runs": len(runs),
        "overall": aggregate(runs),
        "per_puzzle": {
            puzzle: aggregate([run for run in runs if run["puzzle"] == puzzle])
            for puzzle in per_puzzle
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the assembly algorithm against ground truth puzzles."
    )
    parser.add_argument("config", type=str, help="Configuration file")
    parser.add_argument(
        "puzzles_path",
        type=str,
        help="Path to the file listing the puzzles which should be used for "
        "evaluation",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the directory where the puzzles are stored",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to assemble each puzzle (default: 1). The assembly "
        "search is stochastic, so several runs are needed to see its spread.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the first run of each puzzle; further repeats use "
        "consecutive seeds (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write per-run CSV and aggregated JSON results into. "
        "If not given, results are only printed.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="eval",
        help="Name for this evaluation, used for the result filenames "
        "(default: eval)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.puzzles_path) as f:
        puzzle_dirs = [
            os.path.join(args.dataset_path, line.strip()) for line in f if line.strip()
        ]

    runs = []
    for puzzle_dir in puzzle_dirs:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            metrics = evaluate_puzzle(config, puzzle_dir, seed)
            run = {"puzzle": os.path.basename(puzzle_dir), "seed": seed, **metrics}
            runs.append(run)
            print(run["puzzle"], f"seed={seed}", metrics)

    print()
    print(",".join(["puzzle", "seed"] + METRIC_NAMES))
    for run in runs:
        print(",".join(str(run[key]) for key in ["puzzle", "seed"] + METRIC_NAMES))

    print()
    print(f"aggregated over {len(runs)} runs of {len(puzzle_dirs)} puzzles:")
    for name, value in aggregate(runs).items():
        print(f"  {name}: {value}")

    if args.output_dir:
        csv_path, json_path = write_results(args.output_dir, args.label, runs)
        print(f"\nwrote {csv_path}\nwrote {json_path}")
