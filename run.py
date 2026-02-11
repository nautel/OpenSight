"""
OpenSight batch processing pipeline with scene-level temporal fusion.

Pipeline per scene:
1. Pass 1: Single-frame detection (region growing + extremal bbox)
2. Pass 2: Temporal fusion (box projection + missed detection recovery)
3. Pass 3: Spatial awareness (Object Bank + augmentation)
4. Pass 4: Size prior filtering (optional)

Usage:
    python -m OpenSight --split train --temporal --spatial --verbose
"""

import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

from .config import (
    CAMERA_NAMES, ANCHOR_SIZES, DEPTH_RANGE,
    REGION_GROWING_PARAMS, MIN_POINTS_FOR_OUTPUT,
    SPATIAL_AWARENESS_CONFIG, CHECKPOINT_INTERVAL,
    get_paths,
)
from .data_loader import OpenSightLoader
from .mask_processor import extract_in_mask_points
from .region_growing import RegionGrowingFilter
from .bbox_fitter import BBoxFitter
from .ground_filter import GroundFilter
from .nms_3d import nms_3d_bboxes, merge_multi_camera_detections
from .temporal_fuser import TemporalFuser
from .spatial_awareness import SizePriorFilter, ObjectBank, SpatialAugmentor
from .output_formatter import NuScenesFormatter, format_results_for_nuscenes


# =============================================================================
# SINGLE OBJECT OPTIMIZATION
# =============================================================================

def optimize_instance(
    lidar_points: np.ndarray,
    instance: Dict,
    full_mask: np.ndarray,
    intrinsic: np.ndarray,
    lidar2cam: np.ndarray,
    image_shape: Tuple[int, int],
    camera: str,
    region_grower: RegionGrowingFilter,
    bbox_fitter: BBoxFitter,
    ground_filter: GroundFilter,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Optimize 3D bbox for a single instance."""
    instance_id = instance.get('instance_id', 0)
    class_name = instance.get('class_name', 'car')
    class_id = instance.get('class_id', 0)
    detection_score = instance.get('score', 0.5)

    # Create instance mask
    instance_mask = (full_mask == instance_id)
    if not np.any(instance_mask):
        return None

    # Step 1: Extract in-mask points
    in_mask_points, point_indices = extract_in_mask_points(
        lidar_points, instance_mask, intrinsic, lidar2cam, image_shape
    )

    rg_params = REGION_GROWING_PARAMS.get(class_name, REGION_GROWING_PARAMS.get('car'))
    min_points = rg_params[2]

    if len(in_mask_points) < min_points:
        return None

    # Step 2: Region growing clustering (paper Section 3.2)
    rg_result = region_grower.filter(in_mask_points, class_name)
    filtered_points = rg_result.points
    clustering_status = rg_result.status

    # Step 3: Ground estimation
    if len(filtered_points) > 0:
        center_estimate = np.median(filtered_points, axis=0)
        ground_z = ground_filter.estimate_ground_z(
            lidar_points, center_estimate[0], center_estimate[1]
        )
    else:
        ground_z = None

    # Step 4: BBox fitting (extremal method + anchor blending)
    bbox_result = bbox_fitter.fit(
        points=filtered_points,
        class_name=class_name,
        ground_z=ground_z,
    )

    center = bbox_result['center']
    size = bbox_result['size']
    yaw = bbox_result['yaw']
    method = bbox_result['method']

    # Step 5: Ground adjustment
    if ground_z is not None:
        center = ground_filter.adjust_bbox_to_ground(
            center=center, size=size, lidar_points=lidar_points,
        )

    # Step 6: Count points inside final bbox
    points_inside = _count_points_in_bbox(filtered_points, center, size, yaw)

    min_output = MIN_POINTS_FOR_OUTPUT.get(class_name, 3)
    if points_inside < min_output:
        return None

    # Compute score
    score = _compute_score(detection_score, points_inside, clustering_status)

    return {
        'bbox_3d_center': center,
        'bbox_3d_size': size,
        'bbox_3d_yaw': yaw,
        'score': score,
        'class_name': class_name,
        'class_id': class_id,
        'camera': camera,
        'points_inside': points_inside,
        'method': f'opensight_rg_{method}',
        'clustering_status': clustering_status,
        'detection_score': detection_score,
    }


def _count_points_in_bbox(
    points: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> int:
    """Count points inside oriented 3D bbox."""
    if len(points) == 0:
        return 0

    rel_points = points[:, :3] - center
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)

    rotated_x = rel_points[:, 0] * cos_yaw + rel_points[:, 1] * sin_yaw
    rotated_y = -rel_points[:, 0] * sin_yaw + rel_points[:, 1] * cos_yaw
    rotated_z = rel_points[:, 2]

    half_size = size / 2

    inside = (
        (np.abs(rotated_x) <= half_size[0]) &
        (np.abs(rotated_y) <= half_size[1]) &
        (np.abs(rotated_z) <= half_size[2])
    )

    return int(np.sum(inside))


def _compute_score(
    detection_score: float,
    points_inside: int,
    clustering_status: str,
) -> float:
    """Compute final confidence score."""
    score = detection_score
    density_bonus = min(0.2, points_inside / 50)

    if clustering_status == 'fallback':
        score *= 0.8

    score = score + density_bonus
    return min(1.0, max(0.01, score))


# =============================================================================
# CAMERA / SAMPLE / SCENE PROCESSING
# =============================================================================

def process_camera(
    lidar_points: np.ndarray,
    cam_data: Dict,
    camera: str,
    region_grower: RegionGrowingFilter,
    bbox_fitter: BBoxFitter,
    ground_filter: GroundFilter,
    image_shape: Tuple[int, int],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Process all instances in one camera."""
    camera_info = cam_data.get('camera_info')
    if camera_info is None:
        return []

    instances = cam_data['metadata'].get('instances', [])
    if len(instances) == 0:
        return []

    intrinsic = camera_info.get('cam2img')
    lidar2cam = camera_info.get('lidar2cam')
    if intrinsic is None or lidar2cam is None:
        return []

    results = []
    for instance in instances:
        result = optimize_instance(
            lidar_points=lidar_points,
            instance=instance,
            full_mask=cam_data['mask'],
            intrinsic=intrinsic,
            lidar2cam=lidar2cam,
            image_shape=image_shape,
            camera=camera,
            region_grower=region_grower,
            bbox_fitter=bbox_fitter,
            ground_filter=ground_filter,
            verbose=verbose,
        )
        if result is not None:
            results.append(result)

    return results


def process_single_sample(
    loader: OpenSightLoader,
    sample_token: str,
    region_grower: RegionGrowingFilter,
    bbox_fitter: BBoxFitter,
    ground_filter: GroundFilter,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Process a single sample: 6 cameras -> NMS."""
    try:
        lidar_points = loader.load_lidar_points(sample_token)
        cameras_data = loader.get_all_cameras_data(sample_token)
        image_shape = loader.get_image_shape()

        results_per_camera = {}
        for camera in CAMERA_NAMES:
            cam_data = cameras_data.get(camera)
            if cam_data is None:
                continue
            cam_results = process_camera(
                lidar_points, cam_data, camera,
                region_grower, bbox_fitter, ground_filter,
                image_shape, verbose,
            )
            results_per_camera[camera] = cam_results

        # Multi-camera NMS
        final_results = merge_multi_camera_detections(results_per_camera)
        return final_results

    except Exception as e:
        if verbose:
            print(f"Error processing {sample_token}: {e}")
        return []


def process_scene(
    loader: OpenSightLoader,
    scene_samples: List[str],
    region_grower: RegionGrowingFilter,
    bbox_fitter: BBoxFitter,
    ground_filter: GroundFilter,
    use_temporal: bool = True,
    use_spatial: bool = True,
    use_size_filter: bool = False,
    verbose: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Process all samples in a scene with temporal fusion and spatial awareness."""
    if len(scene_samples) == 0:
        return {}

    start_time = time.time()

    if verbose:
        print(f"\nProcessing scene with {len(scene_samples)} samples")

    # Pass 1: Single-frame detection
    detections_cache = {}
    for sample_token in scene_samples:
        dets = process_single_sample(
            loader, sample_token, region_grower, bbox_fitter, ground_filter, verbose
        )
        detections_cache[sample_token] = dets

        if verbose:
            print(f"  Sample {sample_token[:8]}...: {len(dets)} detections")

    # Pass 2: Temporal fusion
    if use_temporal:
        temporal_fuser = TemporalFuser(
            loader, nms_func=nms_3d_bboxes, verbose=verbose
        )
        final_results = {}
        for sample_token in scene_samples:
            fused = temporal_fuser.process_sample_with_context(
                sample_token=sample_token,
                detections_cache=detections_cache,
            )
            final_results[sample_token] = fused

        if verbose:
            total_before = sum(len(d) for d in detections_cache.values())
            total_after = sum(len(d) for d in final_results.values())
            print(f"  Temporal fusion: {total_before} -> {total_after} detections")
    else:
        final_results = detections_cache

    # Pass 3: Spatial awareness augmentation
    if use_spatial:
        spatial_cfg = SPATIAL_AWARENESS_CONFIG

        object_bank = ObjectBank(
            min_points=spatial_cfg.get('min_points_in_bank_entry', 5),
            verbose=verbose,
        )
        object_bank.build_from_scene_results(final_results, loader)

        if object_bank.total_entries() > 0:
            augmentor = SpatialAugmentor(
                object_bank=object_bank,
                max_range=spatial_cfg.get('max_range', 70.0),
                min_placement_distance=spatial_cfg.get('min_placement_distance', 5.0),
                max_placement_distance=spatial_cfg.get('max_placement_distance', 60.0),
                max_augmentations=spatial_cfg.get('max_augmentations_per_sample', 10),
                min_overlap_iou=spatial_cfg.get('min_overlap_iou', 0.05),
                score_factor=spatial_cfg.get('augmented_score_factor', 0.7),
                verbose=verbose,
            )

            total_augmented = 0
            for sample_token in scene_samples:
                try:
                    lidar_points = loader.load_lidar_points(sample_token)
                except Exception:
                    continue

                augmented = augmentor.augment_sample(
                    current_detections=final_results[sample_token],
                    lidar_points=lidar_points,
                )

                if len(augmented) > 0:
                    final_results[sample_token].extend(augmented)
                    final_results[sample_token] = nms_3d_bboxes(
                        final_results[sample_token]
                    )
                    total_augmented += len(augmented)

            if verbose and total_augmented > 0:
                print(f"  Spatial augmentation: +{total_augmented} augmented detections")

    # Pass 4: Size filtering (optional)
    if use_size_filter:
        size_filter = SizePriorFilter(verbose=verbose)
        for sample_token in final_results:
            final_results[sample_token] = size_filter.filter(
                final_results[sample_token]
            )

        if verbose:
            total_filtered = sum(len(d) for d in final_results.values())
            print(f"  After size filter: {total_filtered} detections")

    elapsed = time.time() - start_time
    if verbose:
        print(f"  Scene processed in {elapsed:.1f}s")

    return final_results


# =============================================================================
# BATCH PROCESSING + CHECKPOINT
# =============================================================================

def save_checkpoint(results, processed_scenes, output_dir, name):
    """Save checkpoint."""
    checkpoint = {
        'results': results,
        'processed_scenes': processed_scenes,
    }
    checkpoint_path = output_dir / f'{name}_checkpoint.pkl'
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(output_dir, name):
    """Load checkpoint if exists."""
    checkpoint_path = output_dir / f'{name}_checkpoint.pkl'
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Resumed from checkpoint: {len(checkpoint['processed_scenes'])} scenes")
        return checkpoint
    return None


def build_scene_mapping(loader):
    """Build mapping from scene tokens to ordered sample tokens."""
    scene_to_samples = defaultdict(list)
    sample_to_scene = {}

    for idx in range(len(loader)):
        sample_token = loader.get_sample_token(idx)
        scene_info = loader.get_sample_scene_info(sample_token)
        scene_token = scene_info.get('scene_token', '')

        if scene_token:
            scene_to_samples[scene_token].append(sample_token)
            sample_to_scene[sample_token] = scene_token

    # Order samples within each scene by walking prev/next chain
    for scene_token in scene_to_samples:
        samples = scene_to_samples[scene_token]
        if len(samples) <= 1:
            continue

        # Find first sample
        first_sample = None
        for sample in samples:
            scene_info = loader.get_sample_scene_info(sample)
            if not scene_info.get('prev'):
                first_sample = sample
                break

        if first_sample is None:
            continue

        # Walk forward
        ordered = []
        sample_set = set(samples)
        curr = first_sample
        while curr and curr in sample_set:
            ordered.append(curr)
            scene_info = loader.get_sample_scene_info(curr)
            curr = scene_info.get('next', '')

        scene_to_samples[scene_token] = ordered

    return dict(scene_to_samples), sample_to_scene


def run_batch(
    split: str = 'train',
    use_temporal: bool = True,
    use_spatial: bool = True,
    use_size_filter: bool = False,
    start_idx: int = None,
    end_idx: int = None,
    resume: bool = False,
    verbose: bool = False,
    data_root: Optional[str] = None,
    sam3_root: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Run batch processing grouped by scene."""
    paths = get_paths(data_root=data_root, sam3_root=sam3_root, output_dir=output_dir)
    out_dir = paths['output_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader
    loader = OpenSightLoader(
        split=split,
        data_root=str(paths['data_root']),
        sam3_root=str(paths['sam3_root']),
    )

    # Initialize components
    region_grower = RegionGrowingFilter(verbose=verbose)
    bbox_fitter = BBoxFitter()
    ground_filter = GroundFilter()

    # Build scene mapping
    scene_to_samples, sample_to_scene = build_scene_mapping(loader)
    all_scene_tokens = list(scene_to_samples.keys())

    print(f"\nFound {len(all_scene_tokens)} scenes with "
          f"{sum(len(v) for v in scene_to_samples.values())} total samples")

    # Filter by index range if specified
    if start_idx is not None or end_idx is not None:
        target_samples = set()
        actual_end = end_idx if end_idx is not None else len(loader)
        for idx in range(start_idx or 0, min(actual_end, len(loader))):
            target_samples.add(loader.get_sample_token(idx))

        # Find scenes that contain target samples
        target_scenes = set()
        for sample_token in target_samples:
            scene_token = sample_to_scene.get(sample_token)
            if scene_token:
                target_scenes.add(scene_token)

        all_scene_tokens = [s for s in all_scene_tokens if s in target_scenes]
        print(f"Filtered to {len(all_scene_tokens)} scenes containing indices "
              f"[{start_idx or 0}, {end_idx or len(loader)})")

    # Generate output name
    suffix_parts = []
    if use_temporal:
        suffix_parts.append('temporal')
    if use_spatial:
        suffix_parts.append('spatial')
    if use_size_filter:
        suffix_parts.append('sizefilter')
    suffix = '_'.join(suffix_parts) if suffix_parts else 'baseline'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = f'opensight_{suffix}_{split}_{timestamp}'

    # Check for checkpoint
    results = {}
    processed_scenes = set()

    if resume:
        checkpoint = load_checkpoint(out_dir, output_name)
        if checkpoint:
            results = checkpoint['results']
            processed_scenes = set(checkpoint['processed_scenes'])

    # Process scenes
    start_time = time.time()

    for i, scene_token in enumerate(tqdm(all_scene_tokens, desc='Processing scenes')):
        if scene_token in processed_scenes:
            continue

        try:
            scene_samples = scene_to_samples[scene_token]
            scene_results = process_scene(
                loader=loader,
                scene_samples=scene_samples,
                region_grower=region_grower,
                bbox_fitter=bbox_fitter,
                ground_filter=ground_filter,
                use_temporal=use_temporal,
                use_spatial=use_spatial,
                use_size_filter=use_size_filter,
                verbose=verbose,
            )
            results.update(scene_results)
            processed_scenes.add(scene_token)

        except Exception as e:
            print(f"\nError processing scene {scene_token}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results, list(processed_scenes), out_dir, output_name)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Processed scenes: {len(processed_scenes)}")
    print(f"Processed samples: {len(results)}")
    total_dets = sum(len(d) for d in results.values())
    print(f"Total detections: {total_dets}")
    if len(results) > 0:
        print(f"Average detections per sample: {total_dets/len(results):.1f}")
        print(f"Average time per sample: {elapsed/len(results):.2f}s")

    # Collect transforms and save
    sample_transforms = {}
    for sample_token in results.keys():
        try:
            lidar2ego, ego2global = loader.get_transforms(sample_token)
            sample_transforms[sample_token] = (lidar2ego, ego2global)
        except Exception:
            pass

    output_paths = format_results_for_nuscenes(
        results,
        output_dir=out_dir,
        output_name=output_name,
        save_json=True,
        save_pkl=True,
        sample_transforms=sample_transforms,
    )

    # Per-class statistics
    print("\nPer-class detection counts:")
    class_counts = {}
    for dets in results.values():
        for det in dets:
            cls = det.get('class_name', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1

    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    # Temporal statistics
    if use_temporal:
        projected_count = sum(
            1 for dets in results.values()
            for det in dets if det.get('is_projected', False)
        )
        print(f"\nTemporal fusion statistics:")
        print(f"  Projected detections recovered: {projected_count}")
        print(f"  Percentage of total: {100*projected_count/max(1, total_dets):.1f}%")

    print(f"\nOutput files:")
    for fmt, path in output_paths.items():
        print(f"  {fmt.upper()}: {path}")

    return results, output_paths


def main():
    parser = argparse.ArgumentParser(
        description='OpenSight 3D Object Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full run with temporal + spatial awareness
    python -m OpenSight --split train --temporal --spatial --verbose

    # Quick test on subset
    python -m OpenSight --split train --start_idx 0 --end_idx 2 --verbose

    # Baseline without temporal/spatial
    python -m OpenSight --split train

    # Resume from checkpoint
    python -m OpenSight --split train --temporal --spatial --resume

    # Custom paths
    python -m OpenSight --split train --data_root /path/to/nuscenes
        """,
    )

    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--temporal', action='store_true', help='Enable temporal fusion')
    parser.add_argument('--spatial', action='store_true', help='Enable spatial awareness')
    parser.add_argument('--size-filter', action='store_true', help='Enable size prior filtering')
    parser.add_argument('--start_idx', type=int, default=None, help='Start sample index')
    parser.add_argument('--end_idx', type=int, default=None, help='End sample index (exclusive)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--data_root', type=str, default=None, help='Path to NuScenes data')
    parser.add_argument('--sam3_root', type=str, default=None, help='Path to SAM3 masks')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')

    args = parser.parse_args()

    print("=" * 60)
    print("OpenSight 3D Object Detection Pipeline")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Temporal fusion: {args.temporal}")
    print(f"Spatial awareness: {args.spatial}")
    print(f"Size filter: {args.size_filter}")
    print("=" * 60)

    run_batch(
        split=args.split,
        use_temporal=args.temporal,
        use_spatial=args.spatial,
        use_size_filter=args.size_filter,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        resume=args.resume,
        verbose=args.verbose,
        data_root=args.data_root,
        sam3_root=args.sam3_root,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
