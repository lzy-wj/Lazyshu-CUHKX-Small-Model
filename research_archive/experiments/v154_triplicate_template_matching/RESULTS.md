# Results

| System | Fold 0 | Fold 1 | Combined |
| --- | ---: | ---: | ---: |
| V127 | 569 | 636 | **1,205/1,357** |
| Three slots per source script | 569 | 633 | 1,202/1,357 |

The corrected capacity exactly reproduces V127 on Fold 0, showing that the
independent decoder already respects most triplicate structure. It introduces
three losses without a gain on Fold 1 and is therefore not a local promotion.

Unlike local recordings, test rounds contain many partial scripts with unequal
lengths. A separately marked test candidate is generated only for structural
disagreement analysis; it is not automatically eligible for submission.
