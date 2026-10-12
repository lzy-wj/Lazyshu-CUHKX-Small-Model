# Dynamic Person Crop Fold-0 Ablation

## Decision

- Status: complete; fold 0 only.
- Result: rejected by the preregistered acceptance gate.
- Fold 1: not inspected or trained.
- Kaggle: no submission made.

The tested dynamic crop reduced standalone accuracy by 0.77 percentage points, fixed V9 blend accuracy by 1.69 points, and fixed decoder accuracy by 2.15 points (14 samples). It must not advance to fold 1.

## Frozen Comparison

- Train/validation split: fold 0, validation users `[3, 8, 18, 23]`, 2,384/652 clips.
- Seed: 42.
- Input: 32 sampled frames, Depth_Color RGB plus IR grayscale, 4 channels, 128 x 128.
- Visual model: Kinetics-initialized R(2+1)D-34, 30 epochs, batch size 8, BF16.
- Multimodal stage: unchanged skeleton/world-frame IMU branches and Transformer fusion, 20 epochs, batch size 8, BF16.
- Soup: original fixed-crop 16-frame staged checkpoint at 0.25 plus the new dynamic-crop 32-frame staged checkpoint at 0.75.
- Evaluation: original, horizontal flip, center 75% crop, and center 75% crop plus flip.
- Ensemble and decoder: `scripts/evaluate_single_ablation.py` with fixed V9 weights, temperature, X3D-L confidence gate, repeat pooling, environment transition matrix, and beam decoder.

Only the person crop trajectory changed. Frame positions, channels, resolution, sensor inputs, model architecture, optimizer, schedule, augmentation, TTA, ensemble weights, and decoder settings remained fixed.

## Fixed-Crop Audit

The reference cache uses one normalized window for every frame in a clip. That window is the union of the best person detections from up to eight IR frames, with Depth_Color fallback, margin 1.4, and minimum side 0.35. It detects 3,306 clips and falls back to full frame for 135 clips.

The reference cache is `data/processed/r2plus1d_yolo_32_128`. Its shape is `[3441, 32, 4, 128, 128]`, size is 7,216,300,032 bytes, and its failed-frame counts are 3,360 Depth_Color plus 3,424 IR.

## Dynamic-Crop Method

1. Run YOLO11n person detection on each distinct IR image used by the frozen 32 temporal positions.
2. Run the identical detector on Depth_Color only where the corresponding IR position has no person detection.
3. Associate detections across time with a fixed confidence, IoU, and center-continuity score.
4. Linearly interpolate missing centers and log box sizes.
5. Smooth center and log size with the fixed five-frame triangular kernel `[1, 2, 3, 2, 1]`.
6. Apply margin 1.4 and minimum side 0.35, then crop the original sampled RGB and IR frames to 128 x 128.
7. Use the old clip-level window only for the 129 clips with no per-frame detection at all.

## Box Quality

| Metric | Result |
|---|---:|
| Detected temporal positions | 102,404 / 110,112 |
| Detection coverage | 92.9999% |
| IR detection coverage | 92.5213% |
| RGB fallback coverage | 0.4786% |
| Clips with any detection | 3,312 / 3,441 |
| Mean selected confidence | 0.8045 |
| Mean center jitter before / after | 0.007001 / 0.004641 |
| Mean log-size jitter before / after | 0.010024 / 0.006938 |
| Mean / median IoU with fixed window | 0.8725 / 0.9373 |
| Mean dynamic/fixed crop area ratio | 0.8991 |
| Selected boxes fully inside final crop | 99.8535% |

The smoothing reduced mean center jitter by 33.7% and mean log-size jitter by 30.8%. The dynamic cache has exactly the same failed-frame counts and an identical availability vector as the frozen cache, so the comparison did not alter frame sampling or modality coverage.

Full audit: `experiments/dynamic_crop/cache/ablation_dynamic_crop_32_128/box_audit.json`.

## Training

| Stage | Dynamic best | Fixed reference best | Epoch | SwanLab |
|---|---:|---:|---:|---|
| 32f visual initialization | 62.4233% | 65.3374% | 14 | `bd8tppze` |
| 32f staged multimodal | 63.9571% | 65.3374% | 19 | `vjjvyy1x` |

The requested new-cache visual initialization was fully retrained; no old fixed-crop visual initializer was reused. The fixed reference checkpoints record older source hashes than the currently installed root scripts. Effective architecture and all explicit hyperparameters were audited as equal, but this source-version drift is a residual reproducibility limitation of comparison against the frozen historical checkpoint.

## Fold-0 Metrics

| Metric | Dynamic | Fixed V9 baseline | Delta | Correct samples |
|---|---:|---:|---:|---:|
| Standalone raw | 67.0245% | 67.7914% | -0.7669 pp | 437 vs 442 |
| Available-only | 68.7107% | 69.4969% | -0.7862 pp | 636 available clips |
| Fixed V9 blend raw | 69.3252% | 71.0123% | -1.6871 pp | 452 vs 463 |
| Fixed decoder | 74.6933% | 76.8405% | -2.1472 pp | 487 vs 501 |

The output candidate has the same 652 indices and labels as the frozen fold-0 reference. The unified report is `experiments/dynamic_crop/reports/fold0_metrics.json` and the candidate SHA256 is `8db28d025806cadaa62036730d0f931baa91413f62d926b94a800357a32803c5`.

## Acceptance Gate

- Standalone improvement of at least 0.5 pp without decoder regression: false.
- Decoder improvement of at least two samples without raw regression: false.
- Advance to fold 1: **no**.

This result rejects this specific smoothed per-frame crop recipe. It does not establish that every dynamic visual frontend is harmful; the likely failure mode is loss of stable scene/body context and frame-to-frame geometric warping after independently resizing changing boxes.

## Artifacts

- Cache manifest: `experiments/dynamic_crop/cache/ablation_dynamic_crop_32_128/manifest.json`.
- Cache tensor: `experiments/dynamic_crop/cache/ablation_dynamic_crop_32_128/frames.uint8.memmap`, 7,216,300,032 bytes, SHA256 `e19047659d72d0efa7f734d05abdeccba27b1b0589f5be2dadbea47d67ff085c`.
- Tracking data: `experiments/dynamic_crop/cache/ablation_dynamic_crop_32_128/tracking.npz`, 4,536,281 bytes.
- Visual checkpoint records: `experiments/dynamic_crop/checkpoints/ablation_dynamic_crop_visual_f0`.
- Staged checkpoint records: `experiments/dynamic_crop/checkpoints/ablation_dynamic_crop_staged_f0`.
- Soup checkpoint: `experiments/dynamic_crop/checkpoints/ablation_dynamic_crop_soup_f0_w075/best_int8.pt`, 65,383,691 bytes.
- Candidate predictions: `experiments/dynamic_crop/predictions/ablation_dynamic_crop_fold0.npz`, 115,522 bytes.
- Logs: `experiments/dynamic_crop/logs`.
- SwanLab local records: `experiments/dynamic_crop/swanlog`.

## Reproduction Commands

All GPU work is serialized through `outputs/logs/gpu-7.lock` by the run scripts.

```bash
bash experiments/dynamic_crop/run_cache.sh detect
bash experiments/dynamic_crop/run_cache.sh cache
bash experiments/dynamic_crop/run_train_fold0.sh
bash experiments/dynamic_crop/run_evaluate_fold0.sh
```
