# V52 Domain-Conditioned Logit-Residual Distillation

V52 extends V51 with two changes that add no backbone computation:

1. The residual Conv1d sees both the V24 pre-adaptation and post-adaptation
   log-probabilities (`80` input channels).
2. Independent residuals are fit for each `(paired/unmatched, environment)`
   domain, matching the already validated V32 domain partition.

Each residual is a zero-initialized `80→40` kernel-1 convolution trained from
the frozen V26 system-teacher pseudo labels. V51's retention `0.25`, 300 steps,
optimizer, V32 gate, and pair-aware decoder are unchanged. The method is
transductive: each target split fits its residuals from model-generated pseudo
labels without target ground truth.

