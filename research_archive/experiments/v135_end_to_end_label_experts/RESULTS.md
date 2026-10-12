# Results

| Expert | Fold 0 conditional | Fold 1 conditional |
| --- | ---: | ---: |
| V127 desk incumbent | 94/145 | 75/103 |
| V135 desk | 80/145 | 70/103 |
| V127 interaction incumbent | 284/388 | 322/393 |
| V135 interaction | 246/388 | 283/393 |

The subset-only experts lose the boundary against omitted labels and remain
weaker even when evaluated conditionally inside their own family. Every
nonzero family-mass fusion loses on Fold 0, so formal selection returns the
unchanged V127 control at 1,205/1,357.

V136 keeps all 40 labels during training and emphasizes hard families only
through loss weighting.
