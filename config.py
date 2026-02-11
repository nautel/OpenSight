"""
Configuration for OpenSight 3D Object Detection.

Self-contained configuration with no external dependencies.
All paths configurable via environment variables or function arguments.

Reference: "OpenSight: A Simple Open-Vocabulary Framework for LiDAR-Based
Object Detection" - Hu Zhang et al., ECCV 2024
"""

import os
from pathlib import Path
from typing import Optional


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

def get_project_root() -> Path:
    """Get project root, defaults to two levels up from this file."""
    return Path(os.environ.get(
        'OPENSIGHT_PROJECT_ROOT',
        str(Path(__file__).parent.parent)
    ))


def get_paths(
    data_root: Optional[str] = None,
    sam3_root: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Get all paths with env-var fallbacks.

    Priority: explicit arg > env var > default relative to project root.

    Args:
        data_root: Path to NuScenes data directory
        sam3_root: Path to SAM3 pre-generated masks
        output_dir: Path to output directory

    Returns:
        Dict with keys: data_root, sam3_root, output_dir,
                        nuscenes_info_train, nuscenes_info_val,
                        sam3_index_train, sam3_index_val
    """
    project_root = get_project_root()

    if data_root:
        dr = Path(data_root)
    elif os.environ.get('NUSCENES_ROOT'):
        dr = Path(os.environ['NUSCENES_ROOT'])
    else:
        pkg_data = Path(__file__).parent / 'data' / 'nuscenes'
        root_data = project_root / 'data' / 'nuscenes'
        dr = pkg_data if pkg_data.exists() else root_data

    if sam3_root:
        sr = Path(sam3_root)
    elif os.environ.get('SAM3_MASK_ROOT'):
        sr = Path(os.environ['SAM3_MASK_ROOT'])
    else:
        candidates = [
            project_root / 'data' / 'sam3_masks',
            Path(__file__).parent / 'data' / 'sam3_masks',
            project_root / 'GEN_MASK_NUSCENCES_SAM',
        ]
        sr = next((c for c in candidates if c.exists()), candidates[0])

    od = Path(output_dir or os.environ.get(
        'OPENSIGHT_OUTPUT', str(project_root / 'output')
    ))

    return {
        'data_root': dr,
        'sam3_root': sr,
        'output_dir': od,
        'nuscenes_info_train': dr / 'nuscenes_infos_train.pkl',
        'nuscenes_info_val': dr / 'nuscenes_infos_val.pkl',
        'sam3_index_train': sr / 'train' / 'index.pkl',
        'sam3_index_val': sr / 'val' / 'index.pkl',
    }


# =============================================================================
# CAMERA NAMES
# =============================================================================

CAMERA_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

# =============================================================================
# NUSCENES CLASSES
# =============================================================================

NUSCENES_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(NUSCENES_CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(NUSCENES_CLASSES)}

# =============================================================================
# NuScenes Attribute Mapping
# =============================================================================

DEFAULT_ATTRIBUTES = {
    'car': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'trailer': 'vehicle.parked',
    'bus': 'vehicle.parked',
    'construction_vehicle': 'vehicle.parked',
    'bicycle': 'cycle.without_rider',
    'motorcycle': 'cycle.without_rider',
    'pedestrian': 'pedestrian.standing',
    'traffic_cone': '',
    'barrier': '',
}

# =============================================================================
# ANCHOR SIZES: [l, w, h] per class, [min, mean, max]
# =============================================================================

ANCHOR_SIZES = {
    'car': [
        [4.36, 1.87, 1.64],
        [4.63, 1.97, 1.74],
        [4.90, 2.07, 1.84],
    ],
    'truck': [
        [5.46, 2.25, 2.48],
        [6.93, 2.51, 2.84],
        [8.40, 2.78, 3.20],
    ],
    'construction_vehicle': [
        [5.02, 2.29, 2.77],
        [6.37, 2.85, 3.19],
        [7.72, 3.41, 3.61],
    ],
    'bus': [
        [9.68, 2.72, 3.18],
        [11.15, 2.93, 3.44],
        [12.62, 3.14, 3.70],
    ],
    'trailer': [
        [8.38, 2.60, 3.38],
        [10.24, 2.87, 3.87],
        [12.10, 3.14, 4.36],
    ],
    'barrier': [
        [0.40, 2.00, 0.80],
        [0.54, 2.59, 0.96],
        [0.70, 3.20, 1.15],
    ],
    'motorcycle': [
        [1.69, 0.65, 1.27],
        [1.95, 0.74, 1.41],
        [2.21, 0.83, 1.55],
    ],
    'bicycle': [
        [1.59, 0.51, 1.20],
        [1.76, 0.60, 1.44],
        [1.93, 0.69, 1.68],
    ],
    'pedestrian': [
        [0.60, 0.55, 1.54],
        [0.73, 0.67, 1.74],
        [0.86, 0.79, 1.94],
    ],
    'traffic_cone': [
        [0.35, 0.35, 0.95],
        [0.41, 0.41, 1.07],
        [0.47, 0.47, 1.19],
    ],
}

# =============================================================================
# 3D NMS PARAMETERS
# =============================================================================

NMS_IOU_THRESHOLD = 0.5

CLASS_NMS_IOU_THRESHOLDS = {
    'car': 0.5,
    'truck': 0.5,
    'bus': 0.5,
    'trailer': 0.5,
    'construction_vehicle': 0.5,
    'motorcycle': 0.35,
    'bicycle': 0.35,
    'pedestrian': 0.4,
    'traffic_cone': 0.3,
    'barrier': 0.4,
}

# =============================================================================
# DEPTH RANGE FOR POINT EXTRACTION
# =============================================================================

DEPTH_RANGE = (0.5, 80.0)

# =============================================================================
# REGION GROWING CONFIGURATION (Paper Section 3.2)
# =============================================================================

# Class-specific region growing parameters: [radius, unused, min_cluster]
REGION_GROWING_PARAMS = {
    'car': [1.0, 0.5, 5],
    'truck': [1.5, 0.5, 4],
    'bus': [2.0, 0.5, 4],
    'trailer': [2.0, 0.5, 4],
    'construction_vehicle': [1.5, 0.5, 4],
    'motorcycle': [0.6, 0.4, 3],
    'bicycle': [0.5, 0.4, 3],
    'pedestrian': [0.4, 0.3, 3],
    'traffic_cone': [0.3, 0.2, 2],
    'barrier': [0.8, 0.3, 3],
}

# =============================================================================
# BBOX FITTING CONFIGURATION
# =============================================================================

BBOX_CONFIG = {
    'anchor_weight': 0.6,
    'use_extremal_method': True,
}

# =============================================================================
# GROUND FILTERING CONFIGURATION
# =============================================================================

GROUND_CONFIG = {
    'use_ransac': True,
    'ransac_threshold': 0.15,
    'ransac_iterations': 100,
    'ground_z_offset': 0.1,
    'default_ground_z': -1.7,
}

# =============================================================================
# MINIMUM POINTS THRESHOLDS
# =============================================================================

MIN_POINTS_FOR_OUTPUT = {
    'car': 5,
    'truck': 5,
    'bus': 5,
    'trailer': 5,
    'construction_vehicle': 5,
    'motorcycle': 4,
    'bicycle': 3,
    'pedestrian': 3,
    'traffic_cone': 2,
    'barrier': 3,
}

# =============================================================================
# TEMPORAL FUSION CONFIGURATION (Paper Section 3.2 - Temporal Awareness)
# =============================================================================

TEMPORAL_CONFIG = {
    'enabled': True,
    'use_prev_frame': True,
    'use_next_frame': True,
    'iou_threshold': 0.1,
    'distance_threshold': 3.0,
    'projected_score_factor': 0.9,
    'min_projected_score': 0.1,
    'max_time_gap': 0.6,
    'require_same_class': True,
    'scenario_b_enabled': True,
    'scenario_b_distance_from_ego': 30.0,
    'class_overrides': {
        'truck': {'iou_threshold': 0.08, 'distance_threshold': 4.0},
        'bus': {'iou_threshold': 0.08, 'distance_threshold': 5.0},
        'trailer': {'iou_threshold': 0.08, 'distance_threshold': 5.0},
        'pedestrian': {'iou_threshold': 0.15, 'distance_threshold': 2.0},
        'traffic_cone': {'iou_threshold': 0.2, 'distance_threshold': 1.5},
        'bicycle': {'iou_threshold': 0.15, 'distance_threshold': 2.5},
        'motorcycle': {'iou_threshold': 0.15, 'distance_threshold': 2.5},
    },
}


def get_class_thresholds(class_name: str) -> dict:
    """
    Get IoU and distance thresholds for a specific class.

    Args:
        class_name: NuScenes class name

    Returns:
        Dict with 'iou_threshold' and 'distance_threshold'
    """
    overrides = TEMPORAL_CONFIG.get('class_overrides', {})
    if class_name in overrides:
        return {
            'iou_threshold': overrides[class_name].get(
                'iou_threshold', TEMPORAL_CONFIG['iou_threshold']
            ),
            'distance_threshold': overrides[class_name].get(
                'distance_threshold', TEMPORAL_CONFIG['distance_threshold']
            ),
        }
    return {
        'iou_threshold': TEMPORAL_CONFIG['iou_threshold'],
        'distance_threshold': TEMPORAL_CONFIG['distance_threshold'],
    }


# =============================================================================
# SIZE PRIOR FILTERING CONFIGURATION (Paper Eq. 2)
# =============================================================================

SIZE_PRIOR_CONFIG = {
    'enabled': False,
    'default_tolerance': 0.3,
    'use_anchor_bounds': True,
    'mode': 'reject',
    'class_tolerances': {
        'car': 0.3,
        'truck': 0.4,
        'bus': 0.4,
        'trailer': 0.5,
        'construction_vehicle': 0.5,
        'traffic_cone': 0.25,
        'pedestrian': 0.35,
    },
    'dimension_weights': {
        'length': 1.0,
        'width': 1.0,
        'height': 1.0,
    },
}

# =============================================================================
# SPATIAL AWARENESS CONFIGURATION (Paper Section 3.2 - Spatial Awareness)
# =============================================================================

SPATIAL_AWARENESS_CONFIG = {
    'enabled': True,
    'max_range': 70.0,
    'min_placement_distance': 5.0,
    'max_placement_distance': 60.0,
    'max_augmentations_per_sample': 10,
    'min_overlap_iou': 0.05,
    'augmented_score_factor': 0.7,
    'min_points_in_bank_entry': 5,
}

# =============================================================================
# BATCH PROCESSING
# =============================================================================

CHECKPOINT_INTERVAL = 10
