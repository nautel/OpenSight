"""
Region Growing point filter for 3D object detection.

Paper reference (Section 3.2): "We then apply the region growing algorithm
to isolate the cluster with the densest LiDAR points."

Algorithm:
    1. Build KDTree from spatial coordinates
    2. Find seed = point with most neighbors within class-specific radius
    3. BFS region grow from seed using spatial radius
    4. Fallback: if cluster too small, retry with 1.5x radius or return all
"""

import numpy as np
from dataclasses import dataclass
from collections import deque

from scipy.spatial import cKDTree

from .config import REGION_GROWING_PARAMS


@dataclass
class RegionGrowingResult:
    """Result of region growing filtering."""
    points: np.ndarray           # Filtered points (M, 3)
    point_indices: np.ndarray    # Original indices of filtered points
    status: str                  # 'success', 'fallback', 'insufficient'
    n_input: int                 # Number of input points
    n_output: int                # Number of output points


class RegionGrowingFilter:
    """
    Region growing point filter using spatial KDTree.

    Pipeline:
        1. Build KDTree from points[:, :3]
        2. Find seed = point with most neighbors within radius r
        3. BFS region grow: add neighbors within r
        4. If cluster too small, retry with 1.5x radius
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def filter(
        self,
        points: np.ndarray,
        class_name: str,
    ) -> RegionGrowingResult:
        """
        Filter points using region growing.

        Args:
            points: (N, 3) XYZ points
            class_name: Object class for parameter selection

        Returns:
            RegionGrowingResult with filtered points and metadata
        """
        n_input = len(points)

        # Get class-specific parameters
        params = REGION_GROWING_PARAMS.get(class_name, REGION_GROWING_PARAMS.get('car'))
        radius = params[0]
        min_cluster = params[2]

        # Not enough points to do anything
        if n_input < min_cluster:
            return RegionGrowingResult(
                points=points,
                point_indices=np.arange(n_input),
                status='insufficient',
                n_input=n_input,
                n_output=n_input,
            )

        # Build KDTree on spatial coordinates
        tree = cKDTree(points[:, :3])

        # Find seed: point with most neighbors within radius
        neighbor_counts = tree.query_ball_point(points[:, :3], r=radius, return_length=True)
        seed_idx = int(np.argmax(neighbor_counts))

        # Region grow from seed via BFS
        cluster_mask = self._region_grow(points, tree, seed_idx, radius)
        cluster_indices = np.where(cluster_mask)[0]

        # Fallback: if cluster too small, retry with 1.5x radius
        if len(cluster_indices) < min_cluster:
            expanded_radius = radius * 1.5
            cluster_mask = self._region_grow(points, tree, seed_idx, expanded_radius)
            cluster_indices = np.where(cluster_mask)[0]

        # If still too small, return all points
        if len(cluster_indices) < min_cluster:
            return RegionGrowingResult(
                points=points,
                point_indices=np.arange(n_input),
                status='fallback',
                n_input=n_input,
                n_output=n_input,
            )

        filtered_points = points[cluster_indices]

        return RegionGrowingResult(
            points=filtered_points,
            point_indices=cluster_indices,
            status='success',
            n_input=n_input,
            n_output=len(cluster_indices),
        )

    def _region_grow(
        self,
        points: np.ndarray,
        tree: cKDTree,
        seed_idx: int,
        radius: float,
    ) -> np.ndarray:
        """
        BFS region growing from seed point.

        Args:
            points: (N, 3) points
            tree: KDTree built from points
            seed_idx: Starting point index
            radius: Spatial radius for neighbor search

        Returns:
            Boolean mask (N,) indicating cluster membership
        """
        n = len(points)
        visited = np.zeros(n, dtype=bool)
        in_cluster = np.zeros(n, dtype=bool)

        queue = deque([seed_idx])
        visited[seed_idx] = True
        in_cluster[seed_idx] = True

        while queue:
            current = queue.popleft()

            # Find neighbors within radius
            neighbor_indices = tree.query_ball_point(points[current, :3], r=radius)

            for nb in neighbor_indices:
                if visited[nb]:
                    continue

                visited[nb] = True
                in_cluster[nb] = True
                queue.append(nb)

        return in_cluster
