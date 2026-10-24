"""Script for running the clustering algorithm.

Example
--------
python src/piece_assemble/tools/run_assembly.py /path/to/config

"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from typing import TYPE_CHECKING

from image import np_to_pil
from piece_assemble.cluster import ClusterScorer
from piece_assemble.clustering import Clustering
from piece_assemble.config import load_config
from piece_assemble.feature_extraction.segment import (
    MultiOsculatingCircleFeatureExtractor,
)
from piece_assemble.load import load_images
from piece_assemble.piece import Piece

if TYPE_CHECKING:
    from piece_assemble.cluster import Cluster
    from piece_assemble.feature_extraction.base import FeatureExtractor


def build_pieces(config: dict) -> tuple[list[Piece], FeatureExtractor]:
    """Load a puzzle's pieces and extract their features, exactly as
    `run_prediction` does internally.

    Factored out so that other code (e.g. an evaluation of the pairwise
    matching stage alone, independent of the global assembly it feeds into)
    builds pieces the same way the actual assembly pipeline does, rather than
    risking a second, drifting implementation.

    Parameters
    ----------
    config
        The assembly configuration, as loaded from a config YAML file.

    Returns
    -------
    pieces
        The puzzle's pieces, with features extracted.
    feature_extractor
        The feature extractor used, needed to find pairwise matches between
        the returned pieces.
    """
    img_ids, imgs, masks = load_images(config["img_path"], config["piece"]["scale"])

    feature_extractor = MultiOsculatingCircleFeatureExtractor(**config["descriptor"])

    with Pool(config["clustering"]["n_processes"]) as p:
        pieces = p.starmap(
            Piece.from_image,
            zip(
                img_ids,
                imgs,
                masks,
                [feature_extractor] * len(img_ids),
                [config["piece"]["sigma"]] * len(img_ids),
                [config["piece"]["polygon_approximation_tolerance"]] * len(img_ids),
            ),
        )

    return pieces, feature_extractor


def run_prediction(config: dict) -> list[Cluster]:
    """Run the assembly pipeline for a single puzzle.

    Parameters
    ----------
    config
        The assembly configuration, as loaded from a config YAML file.

    Returns
    -------
    All clusters found by the end of the clustering process, best-scoring first.
    If the assembly succeeded, this is a single cluster containing every piece;
    otherwise it is the set of partial solutions the algorithm ended up with.
    """
    pieces, feature_extractor = build_pieces(config)

    cluster_scorer = ClusterScorer(**config["cluster_scorer"])
    clustering = Clustering(pieces, feature_extractor, cluster_scorer)

    clustering.set_logging(**config["logging"])
    clustering(
        **config["clustering"],
        cluster_config=config["cluster"],
        trusted_cluster_config=config["trusted_cluster"],
    )

    return clustering.clusters  # type: ignore


def save_results(clusters: list[Cluster], output_dir: str, max_saved: int = 5) -> None:
    """Save the top-scoring clusters' assembled images, best first, as
    `result_000.png`, `result_001.png`, etc. Unlike the per-iteration
    diagnostic dumps under `logging.output_images_path`, this is just the
    final outcome, meant for a quick visual check without GT.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, cluster in enumerate(clusters[:max_saved]):
        img = np_to_pil(cluster.draw())
        img.save(os.path.join(output_dir, f"result_{i:03d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piece assemble.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Directory to save the top-scoring clusters' assembled images into "
        "(result_000.png = best). Defaults to "
        "output/results/<img_path's last path component> (output/ is "
        "git-ignored). Pass --no-save-results to skip saving entirely.",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Don't save any result images (overrides the --save-results default).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    clusters = run_prediction(config)

    if not args.no_save_results:
        save_dir = args.save_results or os.path.join(
            "output", "results", os.path.basename(os.path.normpath(config["img_path"]))
        )
        save_results(clusters, save_dir)
        print(f"Saved top results to {save_dir}")
