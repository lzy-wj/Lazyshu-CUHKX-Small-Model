# V60 Results

V60 blends V52 and V57 before the inherited repeat recovery and pair-aware
decoder. Fold 0 searches V57 weights
`0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1`.

Fold 0 selects weight `0`: both `0` and `0.025` score `574/652`, and the
tie-break prefers the simpler zero contribution. The unchanged setting reaches
`627/705` on Fold 1, or `1201/1357` combined.

Because the selected V57 weight is zero and the repeat/decoder path makes no
final test-label change, the candidate is exactly the existing V52 CSV. V60
adds no submission information and is rejected as a new Kaggle submission.
