# Results

## Controlled Comparison

V106 tests whether the target adapters should train only on the domain where
V104 uses them. All architecture, pseudo teachers, epochs, and equal fusion
weights remain fixed.

| Candidate | Fold 0 pair | Fold 1 pair | Combined | Target-weighted |
| --- | ---: | ---: | ---: | ---: |
| V89 | 548 | 622 | 1,170 | 88.8052% |
| V104 all-domain adapters | 550 | 625 | **1,175** | **89.2466%** |
| V106 isolated adapters | 548 | 623 | 1,171 | 89.0168% |

Isolation removes V104's Fold-0 gain and retains only one Fold-1 gain. The
result supports a useful distinction: diverse validation domains regularize
adapter learning and BN estimation, while domain routing should happen only at
inference. V104 remains the promoted candidate.

The complete machine-readable audit is in `outputs/evaluation.json`.
