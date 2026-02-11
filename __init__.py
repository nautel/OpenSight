"""
OpenSight: Open-Vocabulary 3D Object Detection via LiDAR-Camera Fusion.

Non-official implementation of "OpenSight: A Simple Open-Vocabulary Framework
for LiDAR-Based Object Detection" (Hu Zhang et al., ECCV 2024).

Pipeline:
    SAM3 Masks + NuScenes LiDAR -> Point-in-Mask -> Region Growing
    -> Extremal BBox Fitting -> Ground Filter -> 3D NMS
    -> Temporal Fusion -> Spatial Awareness -> Submission JSON
"""

__version__ = '1.0.0'

from .config import (
    NUSCENES_CLASSES,
    ANCHOR_SIZES,
    get_paths,
)
from .data_loader import OpenSightLoader, load_sample_data
from .mask_processor import extract_in_mask_points, extract_objects_from_camera
from .region_growing import RegionGrowingFilter
from .bbox_fitter import BBoxFitter
from .ground_filter import GroundFilter
from .nms_3d import nms_3d_bboxes, merge_multi_camera_detections
from .temporal_fuser import TemporalFuser
from .spatial_awareness import SizePriorFilter, ObjectBank, SpatialAugmentor
from .output_formatter import NuScenesFormatter, format_results_for_nuscenes

__all__ = [
    'NUSCENES_CLASSES',
    'ANCHOR_SIZES',
    'get_paths',
    'OpenSightLoader',
    'load_sample_data',
    'extract_in_mask_points',
    'extract_objects_from_camera',
    'RegionGrowingFilter',
    'BBoxFitter',
    'GroundFilter',
    'nms_3d_bboxes',
    'merge_multi_camera_detections',
    'TemporalFuser',
    'SizePriorFilter',
    'ObjectBank',
    'SpatialAugmentor',
    'NuScenesFormatter',
    'format_results_for_nuscenes',
]
