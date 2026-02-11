"""
3D Bounding Box Fitting with Extremal Method + Anchor Constraints.

Paper reference: Find extremal points per axis, traverse angles with
0.95 boundary shrinkage, select tightest fit.

Key features:
    - Extremal method: ConvexHull + minimum area rectangle (BEV)
    - PCA-based yaw estimation
    - Anchor constraint: Blend tight size with class anchor
    - Size clamping to [anchor_min, anchor_max]
"""

import numpy as np
from scipy.spatial import ConvexHull
from typing import Dict, Any, Optional, Tuple

from .config import BBOX_CONFIG, ANCHOR_SIZES


class BBoxFitter:
    """
    3D Bounding Box Fitter with Extremal Method.

    Pipeline:
        1. Extremal method: ConvexHull + min area rectangle in BEV
        2. PCA-based yaw estimation
        3. Anchor constraint blending
        4. Size clamping to class-specific limits
    """

    def __init__(
        self,
        anchor_weight: float = None,
        use_extremal: bool = None,
    ):
        """
        Initialize bbox fitter.

        Args:
            anchor_weight: Weight for anchor blending (0=tight, 1=anchor)
            use_extremal: Use extremal method
        """
        config = BBOX_CONFIG
        self.anchor_weight = anchor_weight or config['anchor_weight']
        self.use_extremal = use_extremal if use_extremal is not None else config['use_extremal_method']

    def fit(
        self,
        points: np.ndarray,
        class_name: str,
        ground_z: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fit 3D bounding box to points.

        Args:
            points: (N, 3) object points
            class_name: Object class for anchor lookup
            ground_z: Ground plane Z coordinate (optional)

        Returns:
            Dict with center, size, yaw, method
        """
        if len(points) < 3:
            return self._create_fallback_bbox(points, class_name, ground_z)

        # Get anchor sizes
        anchors = ANCHOR_SIZES.get(class_name, ANCHOR_SIZES['car'])
        anchor_min = np.array(anchors[0])
        anchor_mean = np.array(anchors[1])
        anchor_max = np.array(anchors[2])

        # Step 1: Fit tight bbox using extremal method
        if self.use_extremal:
            tight_center, tight_size, yaw = self._fit_extremal(points)
        else:
            tight_center, tight_size = self._fit_simple(points)
            yaw = self._estimate_yaw_pca(points)

        # Step 2: Blend with anchor
        blended_size = self._blend_with_anchor(tight_size, anchor_mean, self.anchor_weight)

        # Step 3: Clamp to anchor limits
        final_size = np.clip(blended_size, anchor_min, anchor_max)

        # Step 4: Adjust center height
        center = tight_center.copy()
        if ground_z is not None:
            center[2] = ground_z + final_size[2] / 2
        else:
            z_min = np.min(points[:, 2])
            z_max = np.max(points[:, 2])
            center[2] = (z_min + z_max) / 2

        return {
            'center': center,
            'size': final_size,
            'yaw': yaw,
            'method': 'extremal' if self.use_extremal else 'simple',
        }

    def _estimate_yaw_pca(self, points: np.ndarray) -> float:
        """Estimate yaw using PCA on XY plane."""
        if len(points) < 3:
            return 0.0

        points_2d = points[:, :2]

        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(2, len(points_2d)))
            pca.fit(points_2d)
            yaw_vec = pca.components_[0]
            return np.arctan2(yaw_vec[1], yaw_vec[0])
        except Exception:
            return 0.0

    def _fit_extremal(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit oriented bbox using ConvexHull + min area rectangle.

        Returns:
            center: (3,) bbox center
            size: (3,) bbox size [l, w, h]
            yaw: orientation angle
        """
        points_2d = points[:, :2]

        if len(points_2d) < 4:
            center, size = self._fit_simple(points)
            return center, size, 0.0

        try:
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]

            # Find minimum area rectangle
            min_area = float('inf')
            best_rect = (np.mean(points_2d, axis=0), np.array([1.0, 1.0]), 0.0)

            for i in range(len(hull_points)):
                edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
                angle = np.arctan2(edge[1], edge[0])

                cos_a, sin_a = np.cos(-angle), np.sin(-angle)
                rotated = np.column_stack([
                    cos_a * hull_points[:, 0] - sin_a * hull_points[:, 1],
                    sin_a * hull_points[:, 0] + cos_a * hull_points[:, 1]
                ])

                min_xy = np.min(rotated, axis=0)
                max_xy = np.max(rotated, axis=0)
                area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])

                if area < min_area and area > 0:
                    min_area = area
                    center_rot = (min_xy + max_xy) / 2
                    size_2d = max_xy - min_xy

                    center_2d = np.array([
                        cos_a * center_rot[0] + sin_a * center_rot[1],
                        -sin_a * center_rot[0] + cos_a * center_rot[1]
                    ])
                    best_rect = (center_2d, size_2d, angle)

            center_2d, size_2d, yaw = best_rect

            # Height from points
            z_min = np.min(points[:, 2])
            z_max = np.max(points[:, 2])
            height = z_max - z_min
            center_z = (z_min + z_max) / 2

            # Ensure length >= width
            if size_2d[0] < size_2d[1]:
                size_2d = size_2d[::-1]
                yaw = yaw + np.pi / 2

            center = np.array([center_2d[0], center_2d[1], center_z])
            size = np.array([size_2d[0], size_2d[1], height])

            # Ensure positive dimensions
            size = np.maximum(size, 0.1)

            return center, size, yaw

        except Exception:
            center, size = self._fit_simple(points)
            return center, size, 0.0

    def _fit_simple(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple axis-aligned bbox fitting."""
        if len(points) == 0:
            return np.zeros(3), np.ones(3)

        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)

        center = (min_pt + max_pt) / 2
        size = max_pt - min_pt

        size = np.maximum(size, 0.1)

        if size[0] < size[1]:
            size[0], size[1] = size[1], size[0]

        return center, size

    def _blend_with_anchor(
        self,
        tight_size: np.ndarray,
        anchor_size: np.ndarray,
        weight: float,
    ) -> np.ndarray:
        """Blend tight size with anchor size."""
        return (1 - weight) * tight_size + weight * anchor_size

    def _create_fallback_bbox(
        self,
        points: np.ndarray,
        class_name: str,
        ground_z: Optional[float],
    ) -> Dict[str, Any]:
        """Create fallback bbox when insufficient points."""
        anchors = ANCHOR_SIZES.get(class_name, ANCHOR_SIZES['car'])
        anchor_mean = np.array(anchors[1])

        if len(points) > 0:
            center = np.median(points, axis=0)
        else:
            center = np.zeros(3)

        if ground_z is not None:
            center[2] = ground_z + anchor_mean[2] / 2

        return {
            'center': center,
            'size': anchor_mean,
            'yaw': 0.0,
            'method': 'fallback',
        }
