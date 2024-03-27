from __future__ import annotations

from functools import cached_property
from itertools import combinations

import shapely

from piece_assemble.geometry import Transformation, get_common_contour_length
from piece_assemble.osculating_circle_descriptor import OsculatingCircleDescriptor


class Cluster:
    def __init__(
        self, pieces: dict[str, tuple[OsculatingCircleDescriptor, Transformation]]
    ) -> None:
        self._pieces = pieces

        self.descriptors = {key: desc for key, (desc, _) in pieces.items()}
        self.transformations = {key: t for key, (_, t) in pieces.items()}

    @cached_property
    def border_length(self) -> int:
        total_length = 0
        for key1, key2 in combinations(self.piece_ids, 2):
            total_length += get_common_contour_length(
                self.descriptors[key1]._contour, self.descriptors[key2]._contour
            )
        return total_length

    @property
    def piece_ids(self) -> set[str]:
        return set(self._pieces.keys())

    def copy(self) -> Cluster:
        new_cluster = Cluster(self._pieces.copy())
        new_cluster.border_length = self.border_length
        return new_cluster

    def add(
        self, descriptor: OsculatingCircleDescriptor, transformation: Transformation
    ) -> None:
        if descriptor.name in self.piece_ids:
            # TODO: Create more meaningful error
            raise ValueError()

        self._pieces[descriptor.name] = (descriptor, transformation)
        self.descriptors[descriptor.name] = descriptor
        self.transformations[descriptor.name] = transformation

        for key in self.piece_ids:
            if key == descriptor.name:
                continue
            self.border_length += get_common_contour_length(
                descriptor._contour, self.descriptors[key]._contour
            )

        # TODO: Update self_intersection and score

    def transform(self, transformation: Transformation) -> Cluster:
        new_pieces = {
            key: (desc, t.compose(transformation))
            for key, (desc, t) in self._pieces.items()
        }
        new_cluster = Cluster(new_pieces)
        new_cluster.border_length = self.border_length
        return new_cluster

    @cached_property
    def self_intersection(self) -> float:
        polygons = [
            shapely.transform(desc._polygon, lambda pol: t.apply(pol))
            for desc, t in self._pieces.values()
        ]
        return max(
            [
                p1.intersection(p2).area / min(p1.area, p2.area)
                for p1, p2 in combinations(polygons, 2)
            ]
        )

    def intersection(self, polygon) -> float:
        polygons = [
            shapely.transform(desc._polygon, lambda pol: t.apply(pol))
            for desc, t in self._pieces.values()
        ]
        return max(
            [p.intersection(polygon).area / min(p.area, polygon.area) for p in polygons]
        )

    def merge(self, other: Cluster) -> Cluster:
        common_keys = self.piece_ids.intersection(other.piece_ids)

        common_key = common_keys.pop()
        cluster1 = self.transform(self.transformations[common_key].inverse())
        cluster2 = other.transform(other.transformations[common_key].inverse())

        for key in common_keys:
            if not cluster1.transformations[key].is_close(
                cluster2.transformations[key]
            ):
                # TODO: more meaningful error
                raise ValueError()

        new_pieces = cluster1._pieces.copy()
        new_pieces.update(cluster2._pieces)
        new_cluster = Cluster(new_pieces)

        return new_cluster

    @cached_property
    def score(self) -> float:
        # TODO: More sensible score
        return len(self._pieces.keys())
