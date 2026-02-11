# OpenSight: Open-Vocabulary 3D Object Detection

Non-official implementation of [OpenSight (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11118.pdf).

> **OpenSight: A Simple Open-Vocabulary Framework for LiDAR-Based Object Detection**
> Hu Zhang et al. | [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11118.pdf) | [Official Code](https://github.com/huzhangcs/OpenSight)

```
SAM3 Masks + NuScenes LiDAR -> Point-in-Mask -> Region Growing -> Extremal BBox
-> Ground Filter -> 3D NMS -> Temporal Fusion -> Spatial Awareness -> Submission JSON
```

## Results (NuScenes v1.0-mini, mini_train)

| Method | mAP | NDS | car | ped | cone | Speed |
|--------|-----|-----|-----|-----|------|-------|
| **OV-SCAN SC-NOD** | **24.40%** | **24.70%** | **28.1%** | **45.4%** | 46.4% | ~27s/sample |
| **OpenSight** | 21.53% | 23.47% | 21.9% | 41.0% | **55.5%** | ~1.3s/sample |

Pre-computed submissions included in `results/submissions/`.

## Algorithm (Paper Section 3.2)

1. **2D Detection + Frustum Extraction**: Camera images -> SAM3 masks -> project LiDAR to image -> extract points in mask (Eq. 1)
2. **Region Growing**: Isolate cluster with densest LiDAR points (spatial proximity, KDTree + BFS)
3. **Extremal BBox Fitting**: ConvexHull + minimum area rectangle, anchor blending, PCA yaw
4. **Temporal Awareness**: Project boxes from frame t+-1 to t via ego-motion. Scenario A: no IoU match -> add missed detection. Scenario B: overlap + distant -> union box
5. **Spatial Awareness**: LLM size priors (Eq. 2) -> Object Bank -> random placement at varying distances with sampling ratio (Eq. 3)

## Setup

```bash
# 1. Clone
git clone https://github.com/nautel/OpenSight.git && cd OpenSight

# 2. Install dependencies
pip install numpy scipy scikit-learn tqdm shapely pyquaternion

# 3. Download NuScenes v1.0-mini (LiDAR only)
#    From: https://www.nuscenes.org/nuscenes
#    Extract so that data/nuscenes/samples/LIDAR_TOP/*.bin exists
#    Info PKL files and SAM3 masks are already included in this repo.

# 4. (Optional) For evaluation
pip install nuscenes-devkit
```

**What's included in this repo (no extra downloads needed besides NuScenes raw data):**
- `data/nuscenes/*.pkl` -- mmdetection3d info files (train + val)
- `data/sam3_masks/` -- Compressed SAM3 masks for all 323 train + 1 val samples (26MB, 500x compressed)

## Quick Start

Run all commands from the **parent directory** of `OpenSight/`:

```bash
# Full run with temporal + spatial awareness (best results)
python -m OpenSight --split train --temporal --spatial --verbose

# Quick test on subset
python -m OpenSight --split train --start_idx 0 --end_idx 2 --verbose

# Baseline without temporal/spatial
python -m OpenSight --split train

# Evaluate pre-computed results
python -m OpenSight.evaluate \
    --result_path results/submissions/opensight_mAP21.53_NDS23.47.json \
    --version v1.0-mini --eval_set mini_train --verbose
```

Custom paths: `--data_root /path/to/nuscenes --sam3_root /path/to/masks --output_dir /path/to/output`

## Package Structure

```
OpenSight/
├── config.py            # Paths, anchors, thresholds, all parameters
├── data_loader.py       # NuScenes + SAM3 mask loading + NuScenes DB
├── mask_processor.py    # LiDAR-to-camera projection + in-mask extraction
├── region_growing.py    # Region growing clustering (paper Section 3.2)
├── bbox_fitter.py       # Extremal bbox + ConvexHull + anchor blending
├── ground_filter.py     # RANSAC ground plane estimation
├── nms_3d.py            # 3D NMS (Shapely BEV IoU)
├── temporal_fuser.py    # Temporal awareness (box projection + missed detection)
├── spatial_awareness.py # Object Bank + Spatial Augmentor + Size Prior Filter
├── output_formatter.py  # NuScenes submission JSON
├── evaluate.py          # NuScenes evaluation wrapper
├── run.py               # CLI entry point + batch pipeline
├── data/
│   ├── nuscenes/        # Info PKLs (included) + LiDAR bins (download)
│   └── sam3_masks/      # Compressed masks (included, 26MB)
└── results/             # Pre-computed submission
```

## Comparison with OV-SCAN

This package is structured identically to [Implement_OVSCAN](https://github.com/nautel/OVSCAN) for side-by-side comparison. Key differences:

| Feature | OpenSight | OV-SCAN |
|---------|-----------|---------|
| Clustering | Region Growing (KDTree + BFS) | DBSCAN depth clustering |
| BBox Fitting | Extremal (ConvexHull + min area rect) | SC-NOD PSO / UltraFast grid search |
| Temporal | Cross-frame projection + missed recovery | None |
| Spatial | Object Bank + augmentation | None |
| Best mAP | 21.53% | 24.40% |
| Speed | ~4.3s/sample | ~27s/sample (PSO) |

## Citation

```bibtex
@inproceedings{zhang2024opensight,
    title={OpenSight: A Simple Open-Vocabulary Framework for LiDAR-Based Object Detection},
    author={Zhang, Hu and others},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2024}
}
```

## Contributors

- [nautel](https://github.com/nautel) - Implementation and experiments
- [Claude Code](https://claude.ai/code) (Anthropic) - Code review, testing, and deployment

## License

[MIT License](LICENSE)

**Disclaimer**: This is a non-official implementation. For the official version, see [huzhangcs/OpenSight](https://github.com/huzhangcs/OpenSight).
