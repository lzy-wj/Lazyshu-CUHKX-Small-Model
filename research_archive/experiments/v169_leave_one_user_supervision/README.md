# V169 Leave-One-User Supervision

The two historical folds each hold out four users. This is leakage-free but
removes about 500 labeled clips that the final full-data model can use. V169
holds out only the dominant unmatched target user while retaining strict
subject isolation:

- Fold 0 checkpoint warm-start, train on every labeled user except user 23.
- Fold 1 checkpoint warm-start, train on every labeled user except user 22.

The experiment measures whether the current unmatched-domain ceiling is a
representation failure or an artifact of overly sparse fold training data.
No validation-user labels enter optimization.

`evaluate.py` also excludes the target user when fitting the environment
transition prior. It reports raw and Radar-batch-decoded V169 predictions,
fixed blends with V71/V23, and gains/losses against frozen V166.
