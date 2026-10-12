# Results

| System | Fold 0 | Fold 1 | Combined |
| --- | ---: | ---: | ---: |
| V127 | 569 | 636 | 1205/1357 |
| V160 Radar batch key | **571** | **651** | **1222/1357** |

V160 gains two rows on Fold 0 and fifteen rows on Fold 1 with one fixed
parameter-free grouping change. Its local accuracy is 90.05%, versus 88.80%
for V127. The actual public score of the test source-coverage candidate is
0.84079, so the fold improvement does not transfer to the public test domain.
V160 is rejected for deployment.
