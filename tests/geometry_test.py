import warnings

import numpy as np
import pytest

from geometry import fit_transform


def test_fit_transform_survives_failed_ransac_fit() -> None:
    """ransac returns no model when it finds no inliers, which used to raise an
    AttributeError instead of falling back to the least squares fit."""
    # Degenerate input: every point is identical, so no model can be fitted.
    points = np.zeros((5, 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformation = fit_transform(points, points, True)

    assert transformation is not None


def test_fit_transform_recovers_a_known_translation() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 1.0]])
    shift = np.array([3.0, -2.0])

    transformation = fit_transform(points, points + shift, True)

    assert transformation.rotation_angle == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose(transformation.translation, shift, atol=1e-6)
