"""
Script for evaluating the assembly algorithm against ground truth puzzles.

usage: eval_assembly.py [-h] config puzzles_path dataset_path

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

import argparse
import os
from copy import deepcopy

from piece_assemble.config import load_config
from piece_assemble.evaluation import cluster_to_transformations, evaluate
from piece_assemble.load import load_puzzle
from piece_assemble.tools.run_assembly import run_prediction

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
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.puzzles_path) as f:
        puzzle_dirs = [
            os.path.join(args.dataset_path, line.strip()) for line in f if line.strip()
        ]

    output_images_path = config["logging"].get("output_images_path")

    eval_results = {}
    for puzzle_dir in puzzle_dirs:
        gt_pieces, neighbors = load_puzzle(puzzle_dir, scale=config["piece"]["scale"])
        gt_transformations = {
            piece.name: piece.transformation for piece in gt_pieces.values()
        }

        puzzle_config = deepcopy(config)
        puzzle_config["img_path"] = puzzle_dir
        if output_images_path:
            puzzle_config["logging"]["output_images_path"] = os.path.join(
                output_images_path, os.path.basename(puzzle_dir)
            )

        predicted = [
            cluster_to_transformations(cluster)
            for cluster in run_prediction(puzzle_config)
        ]
        metrics = evaluate(
            predicted,
            gt_transformations,
            neighbors,
            config["cluster"]["rotation_tol"],
            config["cluster"]["translation_tol"],
        )
        print(puzzle_dir, metrics)
        eval_results[puzzle_dir] = metrics

    metric_names = list(next(iter(eval_results.values())).keys())
    aggregated_metrics = {
        name: sum(m[name] for m in eval_results.values()) / len(eval_results)
        for name in metric_names
    }

    print(",".join(["puzzle"] + metric_names))
    for puzzle_dir, metrics in eval_results.items():
        print(",".join([puzzle_dir] + [str(metrics[name]) for name in metric_names]))
    print(
        ",".join(
            ["aggregated"] + [str(aggregated_metrics[name]) for name in metric_names]
        )
    )
