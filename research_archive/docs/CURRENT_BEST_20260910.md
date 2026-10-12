# Current Best — 2026-09-10

## Verified public reference

- V24 hard-pseudo submission `56140412`: `0.91044` public accuracy.
- CSV: `experiments/v24_full_refit/outputs/submissions/v24_full_warm60_pseudo_pair_aware.csv`.

## Next submission

- V59 teacher-sequence hybrid: `1197/1357 = 88.2093%` local subject-held-out accuracy.
- Asset size: `71,324,325` bytes; permitted CNN/RNN/Transformer components only.
- Test differences: three labels versus V52 and 24 labels versus public V24.
- CSV: `experiments/v24_full_refit/outputs/submissions/v59_teacher_sequence_hybrid.csv`.
- SHA256: `001139324764176f2f0bab0dea1255dca1526c853034f36495de67002ef5a418`.
- Automated quota-safe submission is scheduled for `2026-09-11 08:05 Asia/Shanghai`.

## Active architecture experiment

- V55 adds 6,144 modality-conditioned channel-affine parameters to the shared
  Depth/Thermal/IR 3D CNN.
- Fold 0 uses GPU 0 and SwanLab run `8n3m157g`.
- Fold 1 starts automatically only if the frozen Fold-0 raw or V52-integration
  gate passes.
