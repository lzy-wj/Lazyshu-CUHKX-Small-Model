# V11 Generalization Protocol

## Objective

Improve subject-disjoint generalization without changing the inference architecture, parameter count, visual cache, ensemble, or decoder.

## Frozen Reference

- Fold 0 validation users: `[3, 8, 18, 23]`.
- Candidate branch: the fixed `0.25` 16-frame endpoint plus `0.75` newly trained 32-frame endpoint.
- Fixed V9 fold-0 metrics: standalone `67.7914%`, ensemble raw `71.0123%`, decoded `76.8405%` (`501/652`).
- Training uses the same visual initializer, seed, optimizer, learning-rate schedule, BF16, EMA, augmentations, auxiliary losses, TTA, ensemble weights, and decoder as V9.

## Single-Variable Candidates

1. `control`: all new method arguments are zero. It validates that the refactored training code reproduces the historical endpoint before candidates are interpreted.
2. `structured`: replace the old sensor-mask/no-op-visual policy with symmetric video/skeleton/IMU branch dropout at `p=0.10`. Dropped normalized inputs are zeroed, masks and auxiliary losses agree, and at least one naturally available branch is retained.
3. `contrastive`: add a maximum weight `0.05` cross-user supervised contrastive loss on the fused 192-dimensional class representation. Positives share the action label but come from a different training user; a balanced detached 1024-item queue supplies cross-batch pairs after a 512-item warmup.
4. `conditional_adv`: use action-conditioned linear user heads through gradient reversal, inverse-square-root `(action,user)` weighting, one adversary-only warmup epoch, and maximum coefficient `0.05`. User labels and heads are absent from inference artifacts.
5. `freeze_bn`: keep all backbone weights trainable but freeze the small-batch R(2+1)D BatchNorm running statistics inherited from the visual initializer.
6. `bn_momentum`: replace the backbone BatchNorm update momentum `0.9` with the PyTorch-standard `0.1` while retaining trainable affine parameters and running-stat updates.
7. `class_balance`: apply inverse-square-root training-class weights (`class_balance_power=0.5`) to the unchanged main and auxiliary classification losses.
8. `ema_999`: lengthen EMA smoothing from `0.99` to `0.999` without changing gradients, architecture, or inference cost.
9. `speed_jitter`: apply one synchronized normalized-time scale sampled from `[0.8, 1.2]` to video, skeleton, and IMU for each training clip; clean validation is unchanged.

All three retain the exact 64,955,492-parameter inference model. Only one candidate-specific argument is enabled per run.

## Advancement Gate

- Primary V9 gate: standalone improves by at least `0.5` percentage points without decoded regression, or decoded improves by at least two samples without ensemble-raw regression.
- Generalization evidence: report per-user accuracy, worst-user accuracy, and between-user standard deviation for both standalone and the fixed V9 ensemble.
- Only candidates passing the primary gate advance to fold 1. Combination experiments require independent fold-0 gains first.

## Commands

```bash
bash experiments/generalization_v11/run_fold.sh control 2 0
bash experiments/generalization_v11/run_fold.sh structured 2 0
bash experiments/generalization_v11/run_fold.sh contrastive 3 0
bash experiments/generalization_v11/run_fold.sh conditional_adv 7 0
bash experiments/generalization_v11/run_fold.sh freeze_bn 3 0
bash experiments/generalization_v11/evaluate_fold.sh structured 2 0
```

All artifacts are stored under `experiments/generalization_v11/outputs`; GPU work is guarded by the existing per-GPU lock files.
