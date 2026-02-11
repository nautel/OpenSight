"""
Temporal Awareness: Box projection + missed detection recovery.

Merged from box_projector, missed_detector, and temporal_fuser.

Paper reference (Section 3.2):
- Project boxes from frame t+-1 to t via ego-motion transforms
- Scenario A: no IoU match -> add missed detection with reduced score
- Scenario B: overlap + distant -> NMS + union
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .config import TEMPORAL_CONFIG, get_class_thresholds


# =============================================================================
# BOX PROJECTOR
# =============================================================================

class BoxProjector:
    """
    Project 3D bounding boxes between frames using ego motion transforms.

    Transform chain: source_lidar -> global -> target_lidar
    """

    def __init__(self, data_loader, verbose: bool = False):
        self.data_loader = data_loader
        self.verbose = verbose
        self._transform_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def get_lidar2global(self, sample_token: str) -> np.ndarray:
        lidar2ego, ego2global = self.data_loader.get_transforms(sample_token)
        return ego2global @ lidar2ego

    def compute_transform(
        self,
        source_token: str,
        target_token: str,
    ) -> np.ndarray:
        """Compute transformation from source LiDAR frame to target LiDAR frame."""
        cache_key = (source_token, target_token)
        if cache_key in self._transform_cache:
            return self._transform_cache[cache_key]

        lidar2global_src = self.get_lidar2global(source_token)
        lidar2global_tgt = self.get_lidar2global(target_token)
        global2lidar_tgt = np.linalg.inv(lidar2global_tgt)
        transform = global2lidar_tgt @ lidar2global_src

        self._transform_cache[cache_key] = transform
        return transform

    def project_single_box(
        self,
        center: np.ndarray,
        size: np.ndarray,
        yaw: float,
        transform: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Project a single oriented 3D bounding box using transformation matrix."""
        center_homo = np.array([center[0], center[1], center[2], 1.0])
        center_tgt = (transform @ center_homo)[:3]

        forward_src = np.array([np.cos(yaw), np.sin(yaw), 0.0, 0.0])
        forward_tgt = transform @ forward_src
        forward_tgt = forward_tgt[:2]

        if np.linalg.norm(forward_tgt) > 1e-6:
            yaw_tgt = np.arctan2(forward_tgt[1], forward_tgt[0])
        else:
            yaw_tgt = yaw

        return center_tgt, size.copy(), yaw_tgt

    def project_boxes_to_frame(
        self,
        boxes: List[Dict[str, Any]],
        source_token: str,
        target_token: str,
    ) -> List[Dict[str, Any]]:
        """Project a list of 3D boxes from source frame to target frame."""
        if len(boxes) == 0:
            return []

        transform = self.compute_transform(source_token, target_token)
        projected_boxes = []

        for box in boxes:
            try:
                center = np.array(box['bbox_3d_center'])
                size = np.array(box['bbox_3d_size'])
                yaw = box['bbox_3d_yaw']

                center_proj, size_proj, yaw_proj = self.project_single_box(
                    center, size, yaw, transform
                )

                projected = box.copy()
                projected['bbox_3d_center'] = center_proj.tolist()
                projected['bbox_3d_size'] = size_proj.tolist()
                projected['bbox_3d_yaw'] = float(yaw_proj)
                projected['projected_from'] = source_token
                projected['original_center'] = center.tolist()

                projected_boxes.append(projected)
            except Exception as e:
                if self.verbose:
                    print(f"BoxProjector: Failed to project box: {e}")
                continue

        return projected_boxes


# =============================================================================
# IoU COMPUTATION (lightweight, for temporal matching)
# =============================================================================

def _get_bev_corners(center: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
    """Get 4 BEV corners of an oriented bounding box."""
    l, w, _ = size
    x, y, _ = center

    corners_local = np.array([
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2],
        [-l/2,  w/2],
    ])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw],
    ])

    return corners_local @ rot_matrix.T + np.array([x, y])


def compute_iou_3d(
    center1: np.ndarray, size1: np.ndarray, yaw1: float,
    center2: np.ndarray, size2: np.ndarray, yaw2: float,
) -> float:
    """Approximate 3D IoU using axis-aligned BEV + height overlap."""
    # Height overlap
    z1_min = center1[2] - size1[2] / 2
    z1_max = center1[2] + size1[2] / 2
    z2_min = center2[2] - size2[2] / 2
    z2_max = center2[2] + size2[2] / 2

    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    z_union = max(z1_max, z2_max) - min(z1_min, z2_min)

    if z_union <= 0:
        return 0.0

    height_iou = z_overlap / z_union

    # BEV IoU (axis-aligned approximation)
    corners1 = _get_bev_corners(center1, size1, yaw1)
    corners2 = _get_bev_corners(center2, size2, yaw2)

    min1 = corners1.min(axis=0)
    max1 = corners1.max(axis=0)
    min2 = corners2.min(axis=0)
    max2 = corners2.max(axis=0)

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    if np.any(inter_max <= inter_min):
        return 0.0

    inter_area = (inter_max[0] - inter_min[0]) * (inter_max[1] - inter_min[1])
    area1 = size1[0] * size1[1]
    area2 = size2[0] * size2[1]
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return (inter_area / union_area) * height_iou


def compute_union_box(
    box_a: Dict[str, Any],
    box_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the union bounding box of two boxes."""
    center_a = np.array(box_a['bbox_3d_center'])
    size_a = np.array(box_a['bbox_3d_size'])
    yaw_a = box_a['bbox_3d_yaw']

    center_b = np.array(box_b['bbox_3d_center'])
    size_b = np.array(box_b['bbox_3d_size'])
    yaw_b = box_b['bbox_3d_yaw']

    corners_a = _get_bev_corners(center_a, size_a, yaw_a)
    corners_b = _get_bev_corners(center_b, size_b, yaw_b)

    all_corners = np.vstack([corners_a, corners_b])
    bev_min = all_corners.min(axis=0)
    bev_max = all_corners.max(axis=0)

    union_center_xy = (bev_min + bev_max) / 2
    union_l = bev_max[0] - bev_min[0]
    union_w = bev_max[1] - bev_min[1]

    z_min = min(center_a[2] - size_a[2] / 2, center_b[2] - size_b[2] / 2)
    z_max = max(center_a[2] + size_a[2] / 2, center_b[2] + size_b[2] / 2)
    union_h = z_max - z_min
    union_z = (z_min + z_max) / 2

    score_a = box_a.get('score', 0.5)
    score_b = box_b.get('score', 0.5)
    union_yaw = yaw_a if score_a >= score_b else yaw_b
    higher_score_box = box_a if score_a >= score_b else box_b

    union_box = higher_score_box.copy()
    union_box['bbox_3d_center'] = np.array([union_center_xy[0], union_center_xy[1], union_z])
    union_box['bbox_3d_size'] = np.array([union_l, union_w, union_h])
    union_box['bbox_3d_yaw'] = union_yaw
    union_box['score'] = max(score_a, score_b)
    union_box['is_union'] = True
    union_box['method'] = union_box.get('method', 'unknown') + '_union'

    return union_box


# =============================================================================
# MISSED DETECTOR
# =============================================================================

class MissedDetector:
    """
    Detect missed objects by comparing projected boxes from adjacent frames
    with current frame detections.

    Implements two paper scenarios:
        A: No overlap -> missed detection, add with reduced score
        B: Overlap but far from ego -> compute union box
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.require_same_class = TEMPORAL_CONFIG.get('require_same_class', True)
        self.projected_score_factor = TEMPORAL_CONFIG.get('projected_score_factor', 0.9)
        self.min_projected_score = TEMPORAL_CONFIG.get('min_projected_score', 0.1)
        self.scenario_b_enabled = TEMPORAL_CONFIG.get('scenario_b_enabled', True)
        self.scenario_b_distance = TEMPORAL_CONFIG.get('scenario_b_distance_from_ego', 30.0)

    def find_missed_detections(
        self,
        current_boxes: List[Dict[str, Any]],
        projected_boxes: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
        """
        Find detections from projected boxes that have no match in current boxes.

        Returns:
            Tuple of (missed_list, incomplete_pairs_list)
        """
        if len(projected_boxes) == 0:
            return [], []

        missed = []
        incomplete_pairs = []

        for proj_box in projected_boxes:
            status, matched_box = self._check_detection_status(proj_box, current_boxes)

            if status == 'missed':
                missed_det = proj_box.copy()
                original_score = missed_det.get('score', 0.5)
                missed_det['score'] = max(
                    self.min_projected_score,
                    original_score * self.projected_score_factor
                )
                missed_det['is_projected'] = True
                missed_det['method'] = missed_det.get('method', 'unknown') + '_temporal'
                missed.append(missed_det)

            elif status == 'incomplete' and matched_box is not None:
                incomplete_pairs.append((proj_box, matched_box))

        if self.verbose:
            if len(missed) > 0 or len(incomplete_pairs) > 0:
                print(f"MissedDetector: {len(missed)} missed (A), "
                      f"{len(incomplete_pairs)} incomplete (B) "
                      f"from {len(projected_boxes)} projected boxes")

        return missed, incomplete_pairs

    def _check_detection_status(
        self,
        proj_box: Dict[str, Any],
        current_boxes: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Check detection status for a projected box."""
        proj_class = proj_box.get('class_name', 'unknown')
        thresholds = get_class_thresholds(proj_class)
        iou_thresh = thresholds['iou_threshold']
        dist_thresh = thresholds['distance_threshold']

        proj_center = np.array(proj_box['bbox_3d_center'])
        proj_size = np.array(proj_box['bbox_3d_size'])
        proj_yaw = proj_box['bbox_3d_yaw']

        best_iou = 0.0
        best_match = None
        any_close = False

        for curr_box in current_boxes:
            curr_class = curr_box.get('class_name', 'unknown')

            if self.require_same_class and proj_class != curr_class:
                continue

            curr_center = np.array(curr_box['bbox_3d_center'])
            curr_size = np.array(curr_box['bbox_3d_size'])
            curr_yaw = curr_box['bbox_3d_yaw']

            iou = compute_iou_3d(
                proj_center, proj_size, proj_yaw,
                curr_center, curr_size, curr_yaw
            )

            if iou > best_iou:
                best_iou = iou
                best_match = curr_box

            dist = np.linalg.norm(proj_center - curr_center)
            if dist < dist_thresh:
                any_close = True
                if best_match is None:
                    best_match = curr_box

        if best_iou >= iou_thresh or any_close:
            if self.scenario_b_enabled and best_match is not None:
                ego_dist = np.linalg.norm(proj_center[:2])
                if ego_dist > self.scenario_b_distance:
                    return 'incomplete', best_match

            return 'matched', None
        else:
            return 'missed', None

    def merge_with_current(
        self,
        current_boxes: List[Dict[str, Any]],
        missed_boxes: List[Dict[str, Any]],
        incomplete_pairs: Optional[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Merge current detections with missed detections and union boxes."""
        merged = list(current_boxes)
        merged.extend(missed_boxes)

        if incomplete_pairs:
            replaced_indices = set()

            for proj_box, matched_box in incomplete_pairs:
                for i, box in enumerate(merged):
                    if box is matched_box:
                        replaced_indices.add(i)
                        break

                union = compute_union_box(proj_box, matched_box)
                merged.append(union)

            for idx in sorted(replaced_indices, reverse=True):
                merged.pop(idx)

        return merged


# =============================================================================
# TEMPORAL FUSER (orchestrator)
# =============================================================================

class TemporalFuser:
    """
    Orchestrate temporal awareness for 3D object detection.

    Combines detections from current frame with missed detections
    recovered from adjacent frames.
    """

    def __init__(
        self,
        data_loader,
        nms_func=None,
        verbose: bool = False,
    ):
        self.data_loader = data_loader
        self.verbose = verbose

        if nms_func is None:
            from .nms_3d import nms_3d_bboxes
            self.nms_func = nms_3d_bboxes
        else:
            self.nms_func = nms_func

        self.box_projector = BoxProjector(data_loader, verbose=verbose)
        self.missed_detector = MissedDetector(verbose=verbose)

        self.use_prev = TEMPORAL_CONFIG.get('use_prev_frame', True)
        self.use_next = TEMPORAL_CONFIG.get('use_next_frame', True)
        self.max_time_gap = TEMPORAL_CONFIG.get('max_time_gap', 0.6)

    def get_adjacent_samples(
        self,
        sample_token: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get previous and next sample tokens for a given sample."""
        scene_info = self.data_loader.get_sample_scene_info(sample_token)
        prev_token = scene_info.get('prev', '') or None
        next_token = scene_info.get('next', '') or None

        if self.max_time_gap > 0 and scene_info.get('timestamp'):
            curr_time = scene_info['timestamp']

            if prev_token:
                prev_info = self.data_loader.get_sample_scene_info(prev_token)
                prev_time = prev_info.get('timestamp', 0)
                if abs(curr_time - prev_time) / 1e6 > self.max_time_gap:
                    prev_token = None

            if next_token:
                next_info = self.data_loader.get_sample_scene_info(next_token)
                next_time = next_info.get('timestamp', 0)
                if abs(curr_time - next_time) / 1e6 > self.max_time_gap:
                    next_token = None

        return prev_token, next_token

    def fuse(
        self,
        sample_token: str,
        current_detections: List[Dict[str, Any]],
        adjacent_detections: Dict[str, List[Dict[str, Any]]],
        apply_nms: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fuse current detections with missed detections from adjacent frames."""
        all_missed = []
        all_incomplete = []

        for adj_token, adj_dets in adjacent_detections.items():
            if adj_token is None or adj_dets is None or len(adj_dets) == 0:
                continue

            projected = self.box_projector.project_boxes_to_frame(
                adj_dets,
                source_token=adj_token,
                target_token=sample_token,
            )

            if len(projected) == 0:
                continue

            missed, incomplete_pairs = self.missed_detector.find_missed_detections(
                current_detections,
                projected,
            )

            all_missed.extend(missed)
            all_incomplete.extend(incomplete_pairs)

        merged = self.missed_detector.merge_with_current(
            current_detections,
            all_missed,
            incomplete_pairs=all_incomplete,
        )

        if self.verbose:
            print(f"TemporalFuser: {len(current_detections)} current + "
                  f"{len(all_missed)} missed(A) + {len(all_incomplete)} union(B) "
                  f"= {len(merged)} total")

        if apply_nms and len(merged) > 0:
            merged = self.nms_func(merged)
            if self.verbose:
                print(f"TemporalFuser: {len(merged)} after NMS")

        return merged

    def process_sample_with_context(
        self,
        sample_token: str,
        detections_cache: Dict[str, List[Dict[str, Any]]],
        apply_nms: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process a sample using cached detections from the scene.

        Main entry point for temporal fusion during batch processing.
        """
        current_dets = detections_cache.get(sample_token, [])

        prev_token, next_token = self.get_adjacent_samples(sample_token)

        adjacent_dets = {}

        if self.use_prev and prev_token and prev_token in detections_cache:
            adjacent_dets[prev_token] = detections_cache[prev_token]

        if self.use_next and next_token and next_token in detections_cache:
            adjacent_dets[next_token] = detections_cache[next_token]

        return self.fuse(
            sample_token=sample_token,
            current_detections=current_dets,
            adjacent_detections=adjacent_dets,
            apply_nms=apply_nms,
        )
