# V81 Validation Audit Results

## Direct Evidence

The public leaderboard evaluates 201 of the 405 test clips. V24 is the best
submitted system at `183/201` (`0.91044`). The old OOF ranking is not reliable:

| Model | Old OOF | Public | Public delta vs V24 | Test disagreements vs V24 |
| --- | ---: | ---: | ---: | ---: |
| V24 | 1168/1357 | 183/201 | 0 | 0 |
| V26 | 1170/1357 | 181/201 | -2 | 6 |
| V29 | 1177/1357 | 178/201 | -5 | 16 |
| V59 | 1197/1357 | 180/201 | -3 | 24 |
| V71 | 1201/1357 | 180/201 | -3 | 23 |

V71 gains 33 OOF rows over V24 but loses three public rows. Across nine
submitted systems, old OOF has Pearson correlation `0.397` and Spearman
correlation only `0.077` with public accuracy.

## Distribution Mismatch

The old validation gives roughly one quarter of its rows to each of four
domains. The actual test distribution is:

| Domain | Description | Test rows | Test weight | Validation users |
| --- | --- | ---: | ---: | --- |
| 0 | unmatched environment 0 | 0 | 0.00% | 3, 5 |
| 1 | unmatched environment 1 | 76 | 18.77% | 23, 22 |
| 2 | paired-source environment 0 | 199 | 49.14% | 8, 7 |
| 3 | paired-source environment 1 | 130 | 32.10% | 18, 17 |

Domain 0 therefore contributes about 24% to the old OOF score despite being
absent from test, while domain 2 is heavily underweighted. Reweighting the six
test-relevant validation users by test-domain frequency improves Pearson
correlation to `0.625` and Spearman to `0.298`. Agreement with V24 predictions
has Pearson `0.767` and Spearman `0.655`, but it remains a guardrail rather than
ground truth because V24 still makes 18 public errors.

## V80 Recheck

| Candidate | Old OOF | Relevant-user core | Test-domain weighted |
| --- | ---: | ---: | ---: |
| V23 base | 1054/1357 | 803/1034 | **77.83%** |
| V80 masked | 1055/1357 | 802/1034 | 77.56% |
| V80 encoder only | 1053/1357 | 801/1034 | 77.60% |
| V23/V80 masked blend | **1056/1357** | 803/1034 | 77.70% |

The apparent V80 gain comes from domain 0 and does not survive the test-matched
metric. V80 is demoted and must not replace V23 as the architecture baseline.

## New Protocol

1. Use users `8/18/23` and `7/17/22` as the two primary folds; users `3/5`
   remain safety diagnostics only.
2. Select by test-domain-weighted OOF, then minimum domain accuracy, then old
   eight-user OOF non-regression.
3. Compare full-data test predictions with V24 as a label-free regression
   guardrail. Do not train directly on V24 hard labels and do not infer hidden
   labels from additional leaderboard probes.
4. Submit only candidates whose improvement is supported in domains 1/2/3 and
   whose test changes have a clear model-based justification.
