"""
LiDAR-to-camera projection and in-mask point extraction.

Extracts LiDAR points that project into SAM3 instance masks.
Paper: Eq. 1 - frustum extraction via 2D-to-3D projection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .config import DEPTH_RANGE, REGION_GROWING_PARAMS


def extract_in_mask_points(
    lidar_points: np.ndarray,
    instance_mask: np.ndarray,
    intrinsic: np.ndarray,
    lidar2cam: np.ndarray,
    image_shape: Tuple[int, int],
    depth_range: Tuple[float, float] = DEPTH_RANGE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract LiDAR points that project into an instance mask.

    Args:
        lidar_points: (N, 3) XYZ points in LiDAR frame
        instance_mask: (H, W) boolean mask for this instance
        intrinsic: (3, 3) camera intrinsic matrix (cam2img)
        lidar2cam: (4, 4) LiDAR to camera extrinsic
        image_shape: (H, W) image dimensions
        depth_range: (min_depth, max_depth) for filtering

    Returns:
        (in_mask_points, point_indices): extracted points and their original indices
    """
    if len(lidar_points) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)

    xyz = lidar_points[:, :3]

    # Transform to camera frame
    points_homo = np.hstack([xyz, np.ones((len(xyz), 1))])
    cam_coords = (lidar2cam @ points_homo.T).T[:, :3]

    # Filter behind camera
    in_front = cam_coords[:, 2] > 0.1

    # Project to image
    proj = (intrinsic @ cam_coords[in_front].T).T
    depths = proj[:, 2]
    uv = proj[:, :2] / depths[:, np.newaxis]

    H, W = image_shape

    # Clip to image bounds
    u = np.clip(uv[:, 0].astype(int), 0, W - 1)
    v = np.clip(uv[:, 1].astype(int), 0, H - 1)

    # Check mask
    in_mask = instance_mask[v, u]

    # Check depth range
    d_min, d_max = depth_range
    valid_depth = (depths >= d_min) & (depths <= d_max)

    # Combine filters
    valid = in_mask & valid_depth

    # Map back to original indices
    in_front_indices = np.where(in_front)[0]
    valid_indices = in_front_indices[valid]

    return xyz[valid_indices], valid_indices


def extract_objects_from_camera(
    lidar_points: np.ndarray,
    mask: np.ndarray,
    metadata: Dict,
    camera_info: Dict,
    image_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """
    Extract per-instance points from a single camera.

    Args:
        lidar_points: (N, 3) full LiDAR point cloud
        mask: (H, W) instance segmentation mask
        metadata: Dict with 'instances' list
        camera_info: Dict with 'cam2img' and 'lidar2cam'
        image_shape: (H, W) image dimensions

    Returns:
        List of dicts with 'points', 'indices', 'instance' for each valid instance
    """
    instances = metadata.get('instances', [])
    if len(instances) == 0:
        return []

    intrinsic = camera_info.get('cam2img')
    lidar2cam = camera_info.get('lidar2cam')

    if intrinsic is None or lidar2cam is None:
        return []

    results = []

    for instance in instances:
        instance_id = instance.get('instance_id', 0)
        class_name = instance.get('class_name', 'car')

        instance_mask = (mask == instance_id)
        if not np.any(instance_mask):
            continue

        points, indices = extract_in_mask_points(
            lidar_points, instance_mask, intrinsic, lidar2cam, image_shape
        )

        # Check minimum points
        rg_params = REGION_GROWING_PARAMS.get(class_name, REGION_GROWING_PARAMS.get('car'))
        min_points = rg_params[2]

        if len(points) < min_points:
            continue

        results.append({
            'points': points,
            'indices': indices,
            'instance': instance,
            'instance_mask': instance_mask,
        })

    return results
