"""
RANSAC Ground Plane Estimation and Filtering.

Estimates ground plane from LiDAR points and adjusts bbox bottom to ground level.
"""

import numpy as np
from typing import Tuple, Optional

from .config import GROUND_CONFIG


class GroundFilter:
    """
    RANSAC-based ground plane estimation.

    Pipeline:
        1. RANSAC plane fitting on low-Z points
        2. Estimate ground Z at object location
        3. Adjust bbox bottom to ground level
    """

    def __init__(
        self,
        use_ransac: bool = None,
        ransac_threshold: float = None,
        ransac_iterations: int = None,
        ground_z_offset: float = None,
        default_ground_z: float = None,
    ):
        config = GROUND_CONFIG
        self.use_ransac = use_ransac if use_ransac is not None else config['use_ransac']
        self.ransac_threshold = ransac_threshold or config['ransac_threshold']
        self.ransac_iterations = ransac_iterations or config['ransac_iterations']
        self.ground_z_offset = ground_z_offset or config['ground_z_offset']
        self.default_ground_z = default_ground_z or config['default_ground_z']

    def estimate_ground_z(
        self,
        lidar_points: np.ndarray,
        x_center: float,
        y_center: float,
        search_radius: float = 5.0,
    ) -> float:
        """
        Estimate ground Z at a specific location.

        Args:
            lidar_points: (N, 3+) full LiDAR point cloud
            x_center: X coordinate of interest
            y_center: Y coordinate of interest
            search_radius: Radius to search for ground points

        Returns:
            Estimated ground Z level
        """
        if not self.use_ransac:
            return self._estimate_ground_simple(
                lidar_points, x_center, y_center, search_radius
            )

        if len(lidar_points) == 0:
            return self.default_ground_z

        xyz = lidar_points[:, :3]

        # Filter by XY distance
        distances_xy = np.sqrt(
            (xyz[:, 0] - x_center)**2 +
            (xyz[:, 1] - y_center)**2
        )
        nearby_mask = distances_xy < search_radius

        if np.sum(nearby_mask) < 10:
            nearby_mask = distances_xy < search_radius * 2

        nearby_points = xyz[nearby_mask]

        if len(nearby_points) < 10:
            return self._estimate_ground_simple(
                lidar_points, x_center, y_center, search_radius
            )

        # Filter to low points (likely ground)
        z_vals = nearby_points[:, 2]
        z_threshold = np.percentile(z_vals, 30)
        low_points = nearby_points[z_vals <= z_threshold]

        if len(low_points) < 5:
            return self._estimate_ground_simple(
                lidar_points, x_center, y_center, search_radius
            )

        # RANSAC plane fitting
        plane = self._ransac_plane(low_points)

        if plane is None:
            return self._estimate_ground_simple(
                lidar_points, x_center, y_center, search_radius
            )

        # Compute ground Z at (x_center, y_center)
        a, b, c, d = plane
        if abs(c) > 1e-6:
            ground_z = -(a * x_center + b * y_center + d) / c
        else:
            ground_z = self.default_ground_z

        # Sanity check
        if ground_z > 0 or ground_z < -3.0:
            return self.default_ground_z

        return ground_z + self.ground_z_offset

    def _ransac_plane(
        self,
        points: np.ndarray,
    ) -> Optional[Tuple[float, float, float, float]]:
        """RANSAC plane fitting."""
        n_points = len(points)
        if n_points < 3:
            return None

        best_inliers = 0
        best_plane = None

        for _ in range(self.ransac_iterations):
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            norm_mag = np.linalg.norm(normal)
            if norm_mag < 1e-6:
                continue

            normal = normal / norm_mag
            d = -np.dot(normal, p1)

            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.sum(distances < self.ransac_threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (normal[0], normal[1], normal[2], d)

        return best_plane

    def _estimate_ground_simple(
        self,
        lidar_points: np.ndarray,
        x_center: float,
        y_center: float,
        search_radius: float,
    ) -> float:
        """Simple ground estimation using percentile."""
        if len(lidar_points) == 0:
            return self.default_ground_z

        xyz = lidar_points[:, :3]

        distances_xy = np.sqrt(
            (xyz[:, 0] - x_center)**2 +
            (xyz[:, 1] - y_center)**2
        )
        nearby_mask = distances_xy < search_radius
        nearby_points = xyz[nearby_mask]

        if len(nearby_points) < 5:
            return self.default_ground_z

        ground_z = np.percentile(nearby_points[:, 2], 10)

        if ground_z > 0 or ground_z < -3.0:
            return self.default_ground_z

        return ground_z + self.ground_z_offset

    def adjust_bbox_to_ground(
        self,
        center: np.ndarray,
        size: np.ndarray,
        lidar_points: np.ndarray,
    ) -> np.ndarray:
        """
        Adjust bbox center so bottom is at ground level.

        Args:
            center: (3,) bbox center
            size: (3,) bbox size [l, w, h]
            lidar_points: (N, 3+) full LiDAR point cloud

        Returns:
            Adjusted center (3,)
        """
        ground_z = self.estimate_ground_z(
            lidar_points, center[0], center[1]
        )

        adjusted_center = center.copy()
        adjusted_center[2] = ground_z + size[2] / 2

        return adjusted_center
