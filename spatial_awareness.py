"""
Spatial Awareness: Size Prior Filter + Object Bank + Spatial Augmentor.

Merged from size_prior_filter, object_bank, and spatial_augmentor.

Paper reference (Section 3.2):
- Eq. 2: LLM size priors for filtering invalid detections
- Object Bank: collect validated detections
- Eq. 3: random placement at varying distances with sampling ratio
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from .config import ANCHOR_SIZES, SIZE_PRIOR_CONFIG, SPATIAL_AWARENESS_CONFIG

logger = logging.getLogger(__name__)


# =============================================================================
# SIZE PRIOR FILTER (Paper Eq. 2)
# =============================================================================

class SizePriorFilter:
    """
    Filter detections based on class-specific size priors.

    Two modes:
    1. use_anchor_bounds=True: Use min/max anchor sizes as hard bounds
    2. use_anchor_bounds=False: Use tolerance percentage from mean anchor
    """

    def __init__(
        self,
        tolerance: float = None,
        use_anchor_bounds: bool = None,
        mode: str = None,
        verbose: bool = False,
    ):
        self.tolerance = tolerance if tolerance is not None else SIZE_PRIOR_CONFIG['default_tolerance']
        self.use_anchor_bounds = (
            use_anchor_bounds if use_anchor_bounds is not None
            else SIZE_PRIOR_CONFIG['use_anchor_bounds']
        )
        self.mode = mode if mode is not None else SIZE_PRIOR_CONFIG['mode']
        self.verbose = verbose
        self.class_tolerances = SIZE_PRIOR_CONFIG.get('class_tolerances', {})
        self.dim_weights = SIZE_PRIOR_CONFIG.get('dimension_weights', {})

    def get_size_bounds(self, class_name: str) -> tuple:
        """Get (min_size, max_size) bounds for a class, each [l, w, h]."""
        if class_name not in ANCHOR_SIZES:
            return np.array([0.1, 0.1, 0.1]), np.array([20.0, 10.0, 5.0])

        anchors = ANCHOR_SIZES[class_name]
        min_anchor = np.array(anchors[0])
        mean_anchor = np.array(anchors[1])
        max_anchor = np.array(anchors[2])

        if self.use_anchor_bounds:
            margin = 0.1
            min_size = min_anchor * (1 - margin)
            max_size = max_anchor * (1 + margin)
        else:
            tol = self.class_tolerances.get(class_name, self.tolerance)
            l_tol = tol * self.dim_weights.get('length', 1.0)
            w_tol = tol * self.dim_weights.get('width', 1.0)
            h_tol = tol * self.dim_weights.get('height', 1.0)
            tol_array = np.array([l_tol, w_tol, h_tol])
            min_size = mean_anchor * (1 - tol_array)
            max_size = mean_anchor * (1 + tol_array)

        return min_size, max_size

    def is_valid_size(self, size: np.ndarray, class_name: str) -> bool:
        """Check if a box size is valid for its class."""
        min_size, max_size = self.get_size_bounds(class_name)
        for i in range(3):
            if size[i] < min_size[i] or size[i] > max_size[i]:
                return False
        return True

    def filter(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter detections based on size priors."""
        if len(detections) == 0:
            return []

        result = []
        rejected_count = 0

        for det in detections:
            size = np.array(det.get('bbox_3d_size', [1, 1, 1]))
            class_name = det.get('class_name', 'unknown')

            if self.mode == 'reject':
                if self.is_valid_size(size, class_name):
                    result.append(det)
                else:
                    rejected_count += 1
            else:
                result.append(det)

        if self.verbose and rejected_count > 0:
            print(f"SizePriorFilter: Rejected {rejected_count}/{len(detections)} "
                  f"detections with invalid sizes")

        return result


# =============================================================================
# OBJECT BANK
# =============================================================================

@dataclass
class ObjectBankEntry:
    """A single entry in the object bank."""
    bbox_center: np.ndarray       # (3,) center in LiDAR frame
    bbox_size: np.ndarray         # (3,) [l, w, h]
    bbox_yaw: float
    points: np.ndarray            # (M, 3) LiDAR points inside bbox (local frame)
    class_name: str
    score: float
    original_distance: float      # Distance from ego at detection time
    sample_token: str


class ObjectBank:
    """
    Stores validated detections for spatial awareness augmentation.

    Validates incoming detections against size priors, extracts points
    inside the bbox, and stores for later spatial augmentation.
    """

    def __init__(self, min_points: int = 5, verbose: bool = False):
        self.min_points = min_points
        self.verbose = verbose
        self.size_filter = SizePriorFilter(verbose=False)
        self.bank: Dict[str, List[ObjectBankEntry]] = {}

    def add_detection(
        self,
        detection: Dict[str, Any],
        lidar_points: np.ndarray,
        sample_token: str = '',
    ) -> bool:
        """Validate and add a detection to the bank."""
        class_name = detection.get('class_name', 'car')
        center = np.array(detection['bbox_3d_center'])
        size = np.array(detection['bbox_3d_size'])
        yaw = detection['bbox_3d_yaw']

        if not self.size_filter.is_valid_size(size, class_name):
            return False

        local_points = self._extract_points_in_bbox(
            lidar_points, center, size, yaw
        )

        if len(local_points) < self.min_points:
            return False

        ego_distance = float(np.linalg.norm(center[:2]))

        entry = ObjectBankEntry(
            bbox_center=center.copy(),
            bbox_size=size.copy(),
            bbox_yaw=yaw,
            points=local_points,
            class_name=class_name,
            score=detection.get('score', 0.5),
            original_distance=ego_distance,
            sample_token=sample_token,
        )

        if class_name not in self.bank:
            self.bank[class_name] = []
        self.bank[class_name].append(entry)
        return True

    def build_from_scene_results(
        self,
        scene_results: Dict[str, List[Dict[str, Any]]],
        data_loader,
    ):
        """Build bank from all scene detections."""
        added = 0
        total = 0

        for sample_token, detections in scene_results.items():
            if len(detections) == 0:
                continue

            try:
                lidar_points = data_loader.load_lidar_points(sample_token)
            except Exception as e:
                logger.warning(f"Could not load LiDAR for {sample_token}: {e}")
                continue

            for det in detections:
                total += 1
                if self.add_detection(det, lidar_points, sample_token):
                    added += 1

        if self.verbose:
            print(f"ObjectBank: {added}/{total} detections added to bank")
            for cls, entries in self.bank.items():
                print(f"  {cls}: {len(entries)} entries")

    def get_entries(self, class_name: str) -> List[ObjectBankEntry]:
        return self.bank.get(class_name, [])

    def get_all_classes(self) -> List[str]:
        return list(self.bank.keys())

    def total_entries(self) -> int:
        return sum(len(v) for v in self.bank.values())

    def _extract_points_in_bbox(
        self,
        points: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
        yaw: float,
    ) -> np.ndarray:
        """Extract points inside an oriented 3D bbox, returned in local frame."""
        rel = points[:, :3] - center

        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        rot_x = rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
        rot_y = -rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw
        rot_z = rel[:, 2]

        half = size / 2

        inside = (
            (np.abs(rot_x) <= half[0]) &
            (np.abs(rot_y) <= half[1]) &
            (np.abs(rot_z) <= half[2])
        )

        return np.column_stack([rot_x[inside], rot_y[inside], rot_z[inside]])


# =============================================================================
# SPATIAL AUGMENTOR (Paper Eq. 3)
# =============================================================================

class SpatialAugmentor:
    """
    Augment a sample by placing Object Bank entries at new positions.

    Implements the paper's spatial awareness approach:
    - Random placement at varying distances
    - Point subsampling based on distance ratio (paper Eq. 3)
    - Overlap checking with existing detections
    """

    def __init__(
        self,
        object_bank: ObjectBank,
        max_range: float = 70.0,
        min_placement_distance: float = 5.0,
        max_placement_distance: float = 60.0,
        max_augmentations: int = 10,
        min_overlap_iou: float = 0.05,
        score_factor: float = 0.7,
        verbose: bool = False,
    ):
        self.object_bank = object_bank
        self.max_range = max_range
        self.min_distance = min_placement_distance
        self.max_distance = max_placement_distance
        self.max_augmentations = max_augmentations
        self.min_overlap_iou = min_overlap_iou
        self.score_factor = score_factor
        self.verbose = verbose
        self.rng = np.random.RandomState(42)

    def augment_sample(
        self,
        current_detections: List[Dict[str, Any]],
        lidar_points: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Augment current detections with spatially placed bank entries.

        Returns:
            List of augmented detection dicts (to be added to current)
        """
        if self.object_bank.total_entries() == 0:
            return []

        augmented = []
        all_classes = self.object_bank.get_all_classes()

        if len(all_classes) == 0:
            return []

        existing_boxes = list(current_detections)
        attempts = 0
        max_attempts = self.max_augmentations * 3

        while len(augmented) < self.max_augmentations and attempts < max_attempts:
            attempts += 1

            cls = all_classes[self.rng.randint(len(all_classes))]
            entries = self.object_bank.get_entries(cls)
            if len(entries) == 0:
                continue

            entry = entries[self.rng.randint(len(entries))]

            # Random placement distance and angle
            d_new = self.rng.uniform(self.min_distance, self.max_distance)
            angle = self.rng.uniform(0, 2 * np.pi)

            new_center = np.array([
                d_new * np.cos(angle),
                d_new * np.sin(angle),
                entry.bbox_center[2],
            ])

            new_yaw = entry.bbox_yaw + self.rng.uniform(-0.3, 0.3)

            # Paper Eq. 3: r_sampling = min(1, (d_max - d_new) / (d_max - d_ori))
            d_max = self.max_range
            d_ori = entry.original_distance
            denom = d_max - d_ori
            if denom > 0.1:
                r_sampling = min(1.0, (d_max - d_new) / denom)
            else:
                r_sampling = 1.0
            r_sampling = max(0.1, r_sampling)

            # Subsample points
            n_points = len(entry.points)
            n_keep = max(1, int(n_points * r_sampling))
            if n_keep < n_points:
                keep_idx = self.rng.choice(n_points, n_keep, replace=False)
                local_points = entry.points[keep_idx]
            else:
                local_points = entry.points

            # Check overlap
            if self._has_overlap(new_center, entry.bbox_size, new_yaw, existing_boxes):
                continue

            aug_det = {
                'bbox_3d_center': new_center,
                'bbox_3d_size': entry.bbox_size.copy(),
                'bbox_3d_yaw': new_yaw,
                'score': entry.score * self.score_factor,
                'class_name': entry.class_name,
                'class_id': 0,
                'camera': 'spatial_augmented',
                'points_inside': n_keep,
                'method': 'spatial_augmented',
                'is_spatial_augmented': True,
                'clustering_status': 'spatial',
                'detection_score': entry.score * self.score_factor,
            }

            augmented.append(aug_det)
            existing_boxes.append(aug_det)

        if self.verbose and len(augmented) > 0:
            print(f"SpatialAugmentor: Added {len(augmented)} augmented detections")

        return augmented

    def _has_overlap(
        self,
        center: np.ndarray,
        size: np.ndarray,
        yaw: float,
        existing: List[Dict[str, Any]],
    ) -> bool:
        """Check if a proposed placement overlaps with any existing detection."""
        for det in existing:
            det_center = np.array(det['bbox_3d_center'])
            det_size = np.array(det['bbox_3d_size'])

            dist = np.linalg.norm(center[:2] - det_center[:2])
            r1 = np.linalg.norm(size[:2]) / 2
            r2 = np.linalg.norm(det_size[:2]) / 2
            min_dist = (r1 + r2) * 0.6

            if dist < min_dist:
                return True

        return False
