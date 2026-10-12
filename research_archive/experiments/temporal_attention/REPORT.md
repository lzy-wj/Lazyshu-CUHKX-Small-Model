# Temporal-Attention Fold-0 Ablation

## Decision

**Do not advance to fold 1.** The candidate preserved the fixed repeat-decoder score but reduced both standalone and fixed V9 blend accuracy, so neither acceptance condition passed.

## Controlled Change

- Kept the 32-frame RGB+IR cache, R(2+1)D-34 video encoder, `SkeletonTokens`, `ConvLSTMTokens`, dimensions, optimizer, BF16 precision, EMA, augmentations, seed, split, auxiliary losses, TTA and decoder fixed.
- Replaced only the two-layer Transformer fusion implementation with a two-layer, 6-head, 192-dimensional relative-time-aware attention implementation.
- Assigned each modality independently normalized token-center coordinates in `[0, 1]`.
- Added per-head, per-directed-modality-pair learned relative offset, distance scale and pair bias, allowing video, skeleton and IMU tokens to attend across unequal token rates and different normalized times.
- Trained only the 32-frame staged right endpoint. The final checkpoint uses the protocol's fixed soup: `0.25 * r2_staged_multimodal_f0 + 0.75 * temporal_attention_32f_f0`; right-only temporal parameters come from the trained endpoint.

## Results

| Fold-0 metric | Frozen baseline | Candidate | Delta |
|---|---:|---:|---:|
| Standalone raw | 67.7914% | 67.0245% | -0.7669 pp |
| Available-only | 69.4969% | 68.7107% | -0.7862 pp |
| Fixed V9 blend raw | 71.0123% | 70.3988% | -0.6135 pp |
| Fixed repeat decoder | 76.8405% (501/652) | 76.8405% (501/652) | 0 samples |

- The 32-frame right endpoint selected epoch 2 at 64.8773% no-TTA validation accuracy.
- Against the baseline candidate, 108 standalone predictions changed: 29 errors were corrected and 34 correct predictions regressed.
- V9 confidence gating affected 24 rows under the unified scorer.

## Capacity and Runtime

- Baseline parameters: `64,955,492`.
- Candidate parameters: `64,955,817`.
- Increment: `325` parameters (`0.00050%`).
- Peak allocated GPU memory: `9,699.51 MB`.
- Training elapsed time: `803.14 s` with early stopping at epoch 10.
- SwanLab: <https://swanlab.cn/@shu_123/cuhkx-small-multimodal/runs/9lft1xb0>

## Validation

```bash
conda run --no-capture-output -n Train python -m py_compile \
  experiments/temporal_attention/model.py \
  experiments/temporal_attention/train.py \
  experiments/temporal_attention/evaluate.py \
  experiments/temporal_attention/interpolate.py \
  experiments/temporal_attention/test_model.py

conda run --no-capture-output -n Train \
  python experiments/temporal_attention/test_model.py

bash experiments/temporal_attention/run_fold0.sh
```

The tests cover output shapes, finite values, exact parameter delta and invariance of fused logits to a masked sensor stream. A real 32-frame BF16 GPU forward also passed on locked GPU3 with `[1, 40]` logits and four video tokens.

## Artifacts

- Candidate NPZ: `outputs/predictions/fold0_candidate.npz`
- Unified metrics: `outputs/reports/fold0_unified_metrics.json`
- Training history: `outputs/checkpoints/ablation_temporal_attention_32f_f0/history.jsonl`
- Completion record: `outputs/checkpoints/ablation_temporal_attention_32f_f0/completion.json`
- INT8 fixed soup: `outputs/checkpoints/ablation_temporal_attention_soup_f0_w075/best_int8.pt`

## Interpretation

The learned first-layer normalized time offsets remained small (`mean |offset| = 0.00124`, `max |offset| = 0.00535`), and the global continuous-time gain remained near its `0.1` initialization. This run provides no evidence that explicit asynchronous alignment improves the already temporally normalized cache. The negative raw deltas and unchanged decoder score make the result unsuitable for combination testing under the frozen acceptance gate.
