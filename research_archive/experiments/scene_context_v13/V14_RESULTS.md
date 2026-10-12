# V14 Research Results

## Scores

| Variant | Local subject-held-out | Kaggle public | Public correct |
|---|---:|---:|---:|
| V13 pair-aware baseline | 1161/1357 (85.56%) | 0.89552 | 180/201 |
| V14 fixed epoch soup | 1179/1357 (86.88%) | 0.88557 | 178/201 |
| V14 soup + pseudo adaptation | 1185/1357 (87.32%) | 0.89054 | 179/201 |
| V14 acceptance audit | 1186/1357 (87.40%) | not submitted | n/a |

## Findings

- Averaging epochs 10/14/18 is a repeatable local improvement over a single endpoint.
- Two epochs of legal pseudo-label adaptation on 88,840 head and normalization parameters is best; four epochs overfit.
- Confidence thresholds 0.6 and 0.7, learning rates from 1e-4 to 3e-4, and retention weights 0.25 to 0.5 produce the same final local score neighborhood.
- Expanding adaptation into the last Transformer layer adds parameters without improving the final score.
- The pseudo-adapted submission improves one public sample over the fixed-soup ablation, so pseudo adaptation is helpful; the full-train soup is the source of the public regression.
- The public set has only 201 scored rows. Six changed predictions are enough to move the score by several leaderboard steps, so one public-point regression does not overturn the larger two-fold local result.

## Selection

- Keep V13 submission `56111899` as the current public-leaderboard choice.
- Keep V14 submission `56118201` as the stronger local/private-generalization candidate.
- Do not use the pure fixed-soup submission `56118615` as a final selection.
- Continue only with candidates that add genuine cross-subject diversity; do not tune individual test rows from leaderboard feedback.

## Package

- Reproducible V14 package: `outputs/final_legal_multimodal_v14`.
- Model assets: 99,311,442 bytes.
- Remaining budget: 688,558 bytes.
- Architecture: CNN/RNN/Transformer only; BF16 training; quantized deployment.
