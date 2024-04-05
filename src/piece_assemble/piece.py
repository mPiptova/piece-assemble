from __future__ import annotations

import numpy as np
from more_itertools import flatten
from shapely import geometry
from skimage.filters import rank
from skimage.measure import approximate_polygon
from skimage.morphology import diamond, disk, erosion

from piece_assemble.contours import (
    extract_contours,
    get_osculating_circles,
    get_validity_intervals,
    smooth_contours,
)
from piece_assemble.geometry import (
    extend_interval,
    extend_intervals,
    interval_difference,
    points_dist,
)
from piece_assemble.segment import ApproximatingArc, Segment
from piece_assemble.types import BinImg, NpImage, Points


class Piece:
    def __init__(
        self,
        name: str,
        img: NpImage,
        mask: BinImg,
        sigma: float = 5,
        tol_dist: float = 2.5,
        polygon_approximation_tolerance: float = 3,
        img_mean_window_r: int = 3,
    ) -> None:

        self.name = name
        self.img = img
        self.mask = mask

        # For averaging, use eroded mask for better behavior near contours
        mask_eroded = erosion(self.mask.astype(bool), diamond(1))
        footprint = disk(img_mean_window_r)
        if len(self.img.shape) == 3:
            self.img_avg = (
                np.stack(
                    [
                        rank.mean(self.img[:, :, channel], footprint, mask=mask_eroded)
                        for channel in range(self.img.shape[2])
                    ],
                    axis=2,
                )
                / 255
            )
        else:
            self.img_avg = rank.mean(self.img, footprint, mask=mask_eroded)

        contour = extract_contours(mask)[0]
        self.contour = smooth_contours(contour, sigma)
        self.polygon = geometry.Polygon(
            approximate_polygon(self.contour, polygon_approximation_tolerance)
        )

        # Split curve and extract descriptors
        radii, centers = get_osculating_circles(self.contour)
        self.segments: list[Segment] = approximate_curve_by_circles(
            self.contour, radii, centers, tol_dist
        )

        self.descriptor = np.array(
            [self.segment_descriptor(segment.contour) for segment in self.segments]
        )

    @classmethod
    def segment_descriptor(
        cls, segment: Points, n_points_between: int = 5
    ) -> np.ndarray[float]:
        """Get descriptor of given curve segment.

        Parameters
        ----------
        segment
            2d array of all points representing a contour segment.

        Returns
        -------
        A descriptor of contour segment - an array of 3 2d vectors.
        """
        centroid = segment.mean(axis=0)
        p_start = segment[0]
        p_end = segment[-1]

        # Determine segment rotation
        rot_vector = p_start - p_end
        rot_vector = rot_vector / np.linalg.norm(rot_vector)

        sin_a = rot_vector[0]
        cos_a = rot_vector[1]
        rot_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

        subsegment_len = len(segment) // (n_points_between + 1)
        vectors = [
            p_start,
            *[segment[subsegment_len * (i + 1)] for i in range(n_points_between)],
            p_end,
        ]

        return np.concatenate([(vector - centroid) @ rot_matrix for vector in vectors])

    def get_distances(self, other: Piece) -> np.ndarray:
        desc1 = self.descriptor
        desc2 = other.descriptor

        dist = np.zeros((desc1.shape[0], desc2.shape[0]))

        num_vectors = desc1.shape[1] // 2
        for i in range(num_vectors):
            dist += points_dist(
                desc1[:, i * 2 : (i + 1) * 2],
                -desc2[:, (num_vectors - i - 1) * 2 : (num_vectors - i) * 2],
            )

        return dist / num_vectors

    def get_segment_lengths(self) -> np.ndarray:
        def arc_len(arc: ApproximatingArc):
            extended_interval = extend_interval(arc.interval, len(self.contour))
            return extended_interval[1] - extended_interval[0]

        return np.array([arc_len(arc) for arc in self.segments])

    def filter_small_arcs(self, min_size: float, min_angle: float) -> None:
        """Filter out circle arcs which are too small.

        Parameters
        ----------
        min_size
            Circle arcs with size larger than this number won't be filtered out.
        min_angle
            Circle arcs with the size smaller than `min_size`, but angle larger than
            `min_angle` won't be filtered out.
            The angle is given in radians.
        """

        def is_large_enough(arc: ApproximatingArc) -> bool:
            if (
                np.linalg.norm(
                    self.contour[arc.interval[0]] - self.contour[arc.interval[1]]
                )
                >= min_size
            ):
                return True
            length = len(arc)
            return length >= np.abs(arc.radius) * min_angle

        new_arcs = [arc for arc in self.segments if is_large_enough(arc)]
        if len(new_arcs) == len(self.segments):
            return

        self.segments = new_arcs
        self.descriptor = np.array(
            [self.segment_descriptor(arc.contour) for arc in self.segments]
        )


def approximate_curve_by_circles(
    contour: Points, radii: np.ndarray[float], centers: Points, tol_dist: float
) -> list[ApproximatingArc]:
    """Obtain the curve approximation by osculating circle arcs.

    Parameters
    ----------
    contour
        2d array of all points representing a shape contour.
    radii
        1d array of osculating circle radii.
    centers
        2d array of osculating circle center points.
    tol_dist
        Distance tolerance. If the distance of the contour point and the osculating
        circle is smaller than this number, the point is in the validity interval of
        this circle.

    Returns
    -------
    circles
        I list of circle arc representations. Each element is a tuple
        `(i, validity_interval)`, where `i` is the index of osculating circle and
        `validity_interval` is a range of indexes where the given contour is well
        approximated by this circle.
    """
    validity_intervals = get_validity_intervals_split(contour, radii, centers, tol_dist)
    cycle_length = contour.shape[0]
    validity_intervals_extended = extend_intervals(validity_intervals, cycle_length)
    interval_indexes = np.arange(len(validity_intervals))

    # In each iteration, find the osculating circle with the largest validity interval.
    # Then, update all other intervals and repeat.
    arcs = []
    while True:
        valid_lengths = (
            validity_intervals_extended[:, 1] - validity_intervals_extended[:, 0]
        )
        max_i = np.argmax(valid_lengths)
        length = valid_lengths[max_i]
        if length <= 1:
            break
        validity_interval = validity_intervals[max_i]
        arcs.append((interval_indexes[max_i], validity_interval))
        validity_intervals = np.array(
            [
                interval_difference(r, validity_interval, cycle_length)
                for r in validity_intervals
            ]
        )
        # remove intervals of length 0:
        mask_is_nonzero = validity_intervals[:, 0] != validity_intervals[:, 1]
        validity_intervals = validity_intervals[mask_is_nonzero]
        interval_indexes = interval_indexes[mask_is_nonzero]

        validity_intervals_extended = extend_intervals(validity_intervals, cycle_length)

        if len(validity_intervals_extended) == 0:
            break

    arc_ordering = np.array([c[0] for c in arcs]).argsort()
    return [
        ApproximatingArc(
            arcs[i][1], contour, centers[arcs[i][0]], radii[arcs[i][0]], arcs[i][0]
        )
        for i in arc_ordering
    ]


def get_splitting_points(radii: np.ndarray, min_segment_length: int) -> np.ndarray:
    """Find points where the curve can be split.

    Those points are the global curvature maxima.

    Parameters
    ----------
    radii
        Array of radii.
    min_segment_length
        Minimal length of one part.

    Returns
    -------
    array of indexes where the curve can be split.

    """
    candidate_points = np.argsort(np.abs(radii))
    radii_sorted = np.abs(radii[candidate_points])
    candidate_points = candidate_points[radii_sorted < 15]

    selected_points = []
    while len(candidate_points) > 0:
        new_point = candidate_points[0]
        selected_points.append(new_point)
        candidate_points = candidate_points[
            (np.abs(candidate_points - new_point) >= min_segment_length)
            & (
                (
                    np.minimum(new_point, candidate_points)
                    + len(radii)
                    - np.maximum(new_point, candidate_points)
                )
                >= min_segment_length
            )
        ]

    return np.sort(np.array(selected_points))


def get_validity_intervals_split(
    contour: Points,
    radii: np.ndarray,
    centers: Points,
    tol_dist: float,
    min_segment_length: int = 500,
) -> np.ndarray:
    """Get approximation validity intervals for each osculating circle.

    This function first splits the curve to several separate parts and computes
    the validity intervals for each part separately.

    Parameters
    ----------
    contour
        2d array of all points representing a shape contour.
    radii
        1d array of osculating circle radii.
    centers
        2d array of osculating circle center points.
    tol_dist
        Distance tolerance. If the distance of the contour point and the osculating
        circle is smaller than this number, the point is in the validity interval of
        this circle.
    min_segment_length
        Minimal length of one part.

    Returns
    -------
    validity_intervals
        Array of validity intervals for each osculating circle.
    """
    splitting_points = get_splitting_points(radii, min_segment_length)
    if len(splitting_points) <= 1:
        return get_validity_intervals(contour, radii, centers, tol_dist, True)

    # Shift contour so it starts in the fist one of splitting points
    shift = -splitting_points[0]
    contour = np.roll(contour, shift, axis=0)
    radii = np.roll(radii, shift)
    centers = np.roll(centers, shift, axis=0)
    splitting_points += shift

    segment_ranges = [
        extend_interval(interval, len(radii))
        for interval in zip(splitting_points, np.roll(splitting_points, -1))
    ]
    validity_intervals = list(
        flatten(
            [
                get_validity_intervals(
                    contour[r[0] : r[1]],
                    radii[r[0] : r[1]],
                    centers[r[0] : r[1]],
                    tol_dist,
                    False,
                )
                + r[0]
                for r in segment_ranges
            ]
        )
    )
    validity_intervals = np.roll(validity_intervals, -shift, axis=0)
    return (validity_intervals - shift) % len(radii)