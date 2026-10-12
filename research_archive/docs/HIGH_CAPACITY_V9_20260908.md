# High-Capacity Multimodal V9

## Result

- Best verified Kaggle public score: **0.82089** (`ref=56084496`).
- V9 submission SHA-256: `3d64136e0abc20284fda945c30dc87abe2c2e50207fb6f4ed9833fee2ea84aa2`.
- V9 is byte-identical to the scored V8 decoder submission, so no daily submission slot was consumed.
- Subject-disjoint local decoding: fold 0 **76.8405%**, fold 1 **84.2553%**, weighted **80.6927%**.

## Selected Architecture

1. A calibrated INT8 X3D-S plus statistics baseline.
2. A 32-frame multimodal R(2+1)D with RGB/IR video, skeleton and IMU token fusion.
3. An X3D-L auxiliary encoder over person IR, thermal, skeleton and world-frame IMU.
4. The X3D-L auxiliary is activated only when the main blend confidence is below `0.15`, at weight `0.10`.
5. Repeated rounds use power-mean probability pooling, environment-specific transition priors and beam-100 unique-label assignment.

The X3D-L full-data checkpoint is a `0.25 × epoch24 + 0.75 × epoch30` weight soup. The epoch-24 run was selected before test inference by minimum training-trajectory distance to the epoch-30 run. Original-view and center-87.5%-crop predictions are then blended equally.

## Controlled Findings

| Experiment | Fold 0 | Fold 1 | Decision |
|---|---:|---:|---|
| X3D-M selected soup | 68.25% | 73.05% | Baseline |
| X3D-L selected soup | 69.63% | 73.62% | Better on both folds |
| V8 decoded ensemble | 76.84% | 83.83% | Previous best |
| V9 X3D-L gated decoder | 76.84% | 84.26% | Selected |
| Packed R(2+1)D INT6 | 76.69% | 83.83% | Rejected for accuracy |
| Packed R(2+1)D INT7 | 76.69% | 84.11% | Good fallback, not needed |
| Cross-fold linear stacker | 71.17% | 76.60% | Rejected |

The fold-0 gate search independently selected `confidence < 0.15`, auxiliary weight `0.10`, and temperature `1.0`; applying the same values to fold 1 added three decoded correct samples. Stronger gates improved raw accuracy in places but degraded sequence decoding.

## Compression

- Total model assets: **97,320,445 bytes**, below the decimal 100 MB limit by 2,679,555 bytes.
- Task checkpoints remain INT8. Quantized state tensors are packed into contiguous buffers without changing their values.
- YOLO is metadata-stripped with an exactly matching state dict.
- The SSD preprocessing fallback is per-output-channel INT8; the package also contains the already generated person boxes used by final inference.
- On 192 sampled frames the INT8 SSD retained 99.48% person-presence agreement and 0.956 mean box IoU against FP32.

## Research Boundary

The data contains acquisition identifiers/timestamps that can support label mapping and plausibly explain public scores around 95%. This solution deliberately does not map timestamps, frame IDs, test users or trial identifiers to labels. Under that no-leak policy, **0.82089 is the verified result**; claiming 95% would not be supported by these experiments.

## Artifacts

- Final CSV: `outputs/submissions/final_legal_multimodal_v9.csv`
- Final predictions: `outputs/predictions/final_legal_multimodal_v9.npz`
- Selected config: `configs/v9_selected_ensemble.json`
- X3D-L soup: `outputs/checkpoints/v8_x3dl_soup_full_w075/best_int8_compact.pt`
- SwanLab full runs: `qvo2jpb8`, `cyyq8ooj`, `la4t0l2j`
