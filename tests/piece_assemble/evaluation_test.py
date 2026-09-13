import numpy as np
import pytest

from geometry import Transformation
from piece_assemble.evaluation import (
    _fixed_position_correct_piece_ratio,
    cluster_dict_to_transformations,
    correct_piece_ratio,
    evaluate,
)


@pytest.mark.parametrize(
    "pred_transformations, true_transformations, angle_tol, translation_tol, "
    "expected_result",
    [
        ({}, {}, 0, 0, 1),
        ({"0": {"translation": [0, 0], "rotation_angle": 0}}, {}, 0, 0, 1),
        ({}, {"0": {"translation": [0, 0], "rotation_angle": 0}}, 0, 0, 0),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            0,
            0,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, 0], "rotation_angle": 0}},
            0,
            0,
            0,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, 0], "rotation_angle": 0}},
            0,
            10,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 1}},
            0,
            0,
            0,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 1}},
            1,
            0,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, 0], "rotation_angle": 1}},
            1,
            0,
            0,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, 0], "rotation_angle": 1}},
            1,
            10,
            1,
        ),
        (
            {
                "0": {"translation": [0, 0], "rotation_angle": 0},
                "1": {"translation": [10, 0], "rotation_angle": 1},
            },
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            0,
            0,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {
                "0": {"translation": [0, 0], "rotation_angle": 0},
                "1": {"translation": [10, 0], "rotation_angle": 1},
            },
            0,
            0,
            0.5,
        ),
    ],
)
def test_fixed_position_correct_piece_ratio(
    pred_transformations: dict,
    true_transformations: dict,
    angle_tol: float,
    translation_tol: float,
    expected_result: float,
) -> None:
    pred_transformations = {
        id: Transformation.from_dict(t) for id, t in pred_transformations.items()
    }
    true_transformations = {
        id: Transformation.from_dict(t) for id, t in true_transformations.items()
    }
    assert (
        _fixed_position_correct_piece_ratio(
            pred_transformations, true_transformations, angle_tol, translation_tol
        )
        == expected_result
    )


@pytest.mark.parametrize(
    "pred_transformations, true_transformations, angle_tol, translation_tol, "
    "expected_result",
    [
        # Corner cases
        ({}, {}, 0, 0, 0),
        ({"0": {"translation": [0, 0], "rotation_angle": 0}}, {}, 0, 0, 0),
        ({}, {"0": {"translation": [0, 0], "rotation_angle": 0}}, 0, 0, 0),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            0,
            0,
            1,
        ),
        # One piece
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [0, 0], "rotation_angle": 0.5}},
            0.17,
            21,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, -30], "rotation_angle": 0}},
            0.17,
            21,
            1,
        ),
        (
            {"0": {"translation": [0, 0], "rotation_angle": 0}},
            {"0": {"translation": [10, -30], "rotation_angle": 0.5}},
            0.17,
            21,
            1,
        ),
        # Three pieces, all correct
        # Transformation: {'rotation_angle': 0.1, 'translation': [0, 0]},
        (
            {
                "0": {"translation": [5, 0], "rotation_angle": 0.2},
                "1": {"translation": [10, 10], "rotation_angle": 0.3},
                "2": {"translation": [1, 10], "rotation_angle": 0.8},
            },
            {
                "0": {
                    "rotation_angle": 0.3,
                    "translation": [4.975020826390129, -0.4991670832341408],
                },
                "1": {
                    "rotation_angle": 0.4,
                    "translation": [10.94837581924854, 8.951707486311978],
                },
                "2": {
                    "rotation_angle": 0.9,
                    "translation": [1.9933383317463074, 9.850208236133431],
                },
            },
            0.17,
            21,
            1,
        ),
        # Transformation: {'rotation_angle': 0.6, 'translation': [2, 50]},
        (
            {
                "0": {"translation": [5, 0], "rotation_angle": 0.2},
                "1": {"translation": [10, 10], "rotation_angle": 0.3},
                "2": {"translation": [1, 10], "rotation_angle": 0.8},
            },
            {
                "0": {
                    "rotation_angle": 0.8,
                    "translation": [6.1266780745483915, 47.17678763302482],
                },
                "1": {
                    "rotation_angle": 0.8999999999999999,
                    "translation": [15.899780883047136, 52.60693141514643],
                },
                "2": {
                    "rotation_angle": 1.4,
                    "translation": [8.471760348860032, 57.68871367570175],
                },
            },
            0.17,
            21,
            1,
        ),
        # One off
        # Transformation: {'rotation_angle': 0.6, 'translation': [2, 50]},
        (
            {
                "0": {"translation": [5, 0], "rotation_angle": 0.2},
                "1": {"translation": [10, 10], "rotation_angle": 0.3},
                "2": {"translation": [1, 10], "rotation_angle": 0.8},
            },
            {
                "0": {
                    "rotation_angle": 0.8,
                    "translation": [6.1266780745483915, 47.17678763302482],
                },
                "1": {
                    "rotation_angle": 0.8999999999999999,
                    "translation": [15.899780883047136, 52.60693141514643],
                },
                "2": {"rotation_angle": 2, "translation": [0, 0]},
            },
            0.17,
            21,
            0.666,
        ),
    ],
)
def test_correct_piece_ratio(
    pred_transformations: dict,
    true_transformations: dict,
    angle_tol: float,
    translation_tol: float,
    expected_result: float,
) -> None:
    assembled = cluster_dict_to_transformations(
        {
            "transformed_pieces": [
                {"id": piece_id, "transformation": transformation}
                for piece_id, transformation in pred_transformations.items()
            ]
        }
    )
    ground_truth = cluster_dict_to_transformations(
        {
            "transformed_pieces": [
                {"id": piece_id, "transformation": transformation}
                for piece_id, transformation in true_transformations.items()
            ]
        }
    )

    assert correct_piece_ratio(
        assembled, ground_truth, angle_tol, translation_tol
    ) == pytest.approx(expected_result, 0.01)


GROUND_TRUTH = {
    "0": Transformation(0, np.array([0, 0])),
    "1": Transformation(0, np.array([10, 0])),
    "2": Transformation(0, np.array([20, 0])),
}
NEIGHBORS = [("0", "1"), ("1", "2")]


def test_evaluate_perfect_assembly() -> None:
    predicted = [dict(GROUND_TRUTH)]

    result = evaluate(
        predicted, GROUND_TRUTH, NEIGHBORS, rotation_tol=0.1, translation_tol=1
    )

    assert result["largest_cluster_ratio"] == 1
    assert result["num_of_clusters"] == 1
    assert result["angle_error"] == pytest.approx(0)
    assert result["translation_error"] == pytest.approx(0)


def test_evaluate_partial_assembly() -> None:
    # Pieces "0" and "1" are correctly assembled relative to each other, "2" is
    # left in its own, unrelated position.
    predicted = [
        {
            "0": Transformation(0, np.array([100, 100])),
            "1": Transformation(0, np.array([110, 100])),
            "2": Transformation(1, np.array([0, 0])),
        }
    ]

    result = evaluate(
        predicted, GROUND_TRUTH, NEIGHBORS, rotation_tol=0.1, translation_tol=1
    )

    # Largest correct cluster is {"0", "1"}: 2 out of 3 pieces.
    assert result["largest_cluster_ratio"] == pytest.approx(2 / 3)
    # One correct cluster ({"0", "1"}) plus one leftover piece ("2").
    assert result["num_of_clusters"] == 2


def test_evaluate_empty_prediction() -> None:
    result = evaluate([], GROUND_TRUTH, NEIGHBORS, rotation_tol=0.1, translation_tol=1)

    # Nothing assembled: every piece counts as its own singleton cluster.
    assert result["largest_cluster_ratio"] == pytest.approx(1 / 3)
    assert result["num_of_clusters"] == 3
