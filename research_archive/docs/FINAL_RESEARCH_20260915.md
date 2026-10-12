# Final Research Audit — 2026-09-15

## Outcome

- Verified public incumbent remains `v127_without_0531.csv` at `185/201 = 0.92039`.
- No final-day experiment beats the frozen V166 local system (`1226/1357`) or
  supplies a test-safe improvement over the public incumbent.
- No new Kaggle submission is justified; three daily slots remain unused.

## New Scientific Findings

1. The original local IMU cache omitted magnetometer channels used by the
   official CUHK-X baseline. Restoring the official five-device representation
   improves ExtraTrees across all five user-isolated folds by 50 total rows.
2. Direct frozen-visual integration improves raw V23 by two rows, but does not
   survive the stronger routed sequence system.
3. Statistical residual adapters reproduce a one-row raw gain across two
   learning rates and two seeds. Conservative V166 routing remains unchanged.
4. A zero-initialized five-device attention residual reverses across folds
   (`-3/-4` combined), proving that extra capacity is not the missing factor.
5. The exact official IMU CNN-Transformer reaches only 22.85%/24.68% on held-out
   subjects while fitting training near 99%; its published-style cross-trial
   recipe does not solve this competition's cross-subject shift.

## Community Check

No reproducible high-scoring Small Model solution is public. The September
official repository contributes baseline code rather than a competition-winning
recipe. Recent domain-generalizable HAR work emphasizes temporal binding,
cross-modal contrastive alignment, and online distillation; this repository has
already tested the corresponding asynchronous shift, cross-user/environment
contrastive, and masked-modality distillation families without a stable final
system gain.

## Final Selection

The V166 submission changes 16 test predictions but drops public accuracy from
`185/201` to `178/201`. This is direct evidence that its local structural gains
do not transfer to the test acquisition domain. Select the already-submitted
`v127_without_0531.csv` as the primary final entry. For the second final entry,
select submitted V164: it is only two public rows below the incumbent, changes
seven predictions, and adds eight validation rows without a held-out-user loss.
This gives more useful private diversification than full V127's four-row hedge,
while avoiding V166's disproven sixteen-row risk.
