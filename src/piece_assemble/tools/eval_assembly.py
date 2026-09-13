"""
Script for evaluating the assembly algorithm against ground truth puzzles.

Each puzzle can be assembled several times with different random seeds, since
the assembly search is stochastic and a single run is not representative.

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
from piece_assemble.evaluation import cluster_to_transformations, evaluate
from piece_assemble.load import load_puzzle
from piece_assemble.tools.run_assembly import run_prediction

METRIC_NAMES = [
    "angle_error",
    "translation_error",
    "transformation_error",
    "largest_cluster_ratio",
    "num_of_clusters",
    "runtime_seconds",
]


def evaluate_puzzle(config: dict, puzzle_dir: str, seed: int | None = None) -> dict:
    """Assemble a single puzzle once and score the result against its ground truth.

    Parameters
    ----------
    config
        The assembly configuration. `img_path` is overridden with `puzzle_dir`.
    puzzle_dir
        Directory holding the puzzle's pieces and its pieces.json ground truth.
    seed
        Seed for the assembly search, for repeatable runs.

    Returns
    -------
    The metrics reported by `evaluation.evaluate`, plus the wall-clock runtime.
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

    start = time.monotonic()
    clusters = run_prediction(puzzle_config)
    runtime = time.monotonic() - start

    metrics = evaluate(
        [cluster_to_transformations(cluster) for cluster in clusters],
        gt_transformations,
        neighbors,
        config["cluster"]["rotation_tol"],
        config["cluster"]["translation_tol"],
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
