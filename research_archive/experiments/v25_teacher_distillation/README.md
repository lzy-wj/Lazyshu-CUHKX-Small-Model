# V25 Structured Teacher Distillation

This experiment distills the deployed V13 probability ensemble and its frozen
pair-aware decoder into one V23/V24 student.  It changes only the small
normalization and classification heads, so the resulting adapter remains well
inside the deployment budget when attached to the V24 INT8 model.

## Motivation

- V13 is the current public leader at `180/201`, while V23 is the strongest
  subject-held-out single model.
- Hard pseudo-label adaptation discards the teacher's class similarities.
- The target here is a fixed blend of V13 raw probabilities and smoothed
  pair-aware labels, with KL retention to the unadapted student.

## Frozen screening protocol

1. Build out-of-fold teacher targets without validation-label access.
2. Train two epochs on Fold 0 with `structure_weight=0.20`, `lr=2e-4`, and
   `retention_weight=0.25`.
3. Apply the existing domain blend, pair-aware decoder, and second-stage gate.
4. Inspect Fold 1 only if Fold 0 reaches at least the current V23 second-stage
   score of `566/652` and the raw student does not regress by more than 12 rows.
5. Promote only above the frozen combined system reference of `1190/1357`.

Run after the V24 queue releases its GPU:

```bash
bash experiments/v25_teacher_distillation/run_validation.sh 1
```
