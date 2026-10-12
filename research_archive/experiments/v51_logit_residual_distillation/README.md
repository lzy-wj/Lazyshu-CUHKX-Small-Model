# V51 Transductive Logit-Residual Distillation

## Motivation

V26's multi-model system teacher scores `1195/1357`, but changing the student's
original classification heads retains only `1170/1357`. V51 freezes the entire
V24 model and trains a zero-initialized `1×1` Conv1d residual on its 40 output
log-probabilities. This isolates class-confusion correction from representation
learning and preserves an exact identity initialization.

## Protocol

1. Inputs are V24 heads-only hard-pseudo probabilities.
2. Targets are the frozen V26 system-teacher pseudo labels.
3. A 40-to-40 kernel-1 convolution is trained for 300 full-batch CPU steps with
   AdamW (`lr=0.03`, weight decay `0.01`).
4. Fold 0 selects retention strength from `{0.1, 0.25, 0.5, 1, 2, 4, 8}` by
   decoded accuracy, breaking ties toward stronger retention.
5. The selected strength is applied once to Fold 1, where a new calibrator is
   fit from Fold-1 unlabeled-input teacher predictions; target labels are used
   only for the final audit.
6. V32 domain gating and pair-aware decoding remain frozen.

This is explicitly transductive pseudo-label distillation, like V24's existing
test-time head adaptation. A calibrator trained on Fold 0 and transferred
without refitting does not improve Fold 1; V51 must not be described as a
target-independent calibration model.

The calibrator has 1,640 parameters and adds only a few kilobytes to the V24
CNN/RNN/Transformer package.

