""" Script for running the clustering algorithm.

Example
--------
python src/piece_assemble/tools/run_assembly.py /path/to/config

"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from typing import TYPE_CHECKING

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

    cluster_scorer = ClusterScorer(**config["cluster_scorer"])
    clustering = Clustering(pieces, feature_extractor, cluster_scorer)

    clustering.set_logging(**config["logging"])
    clustering(
        **config["clustering"],
        cluster_config=config["cluster"],
        trusted_cluster_config=config["trusted_cluster"],
    )

    return clustering.clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piece assemble.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_prediction(config)
