# Results

| Variant | Fold 0 | Fold 1 | Combined | Accepted gains/losses | Test changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| V127 | 569 | 636 | 1205/1357 | - | - |
| V164 main | 571 | 642 | **1213/1357** | 8/0 | 7 |
| V164 conservative | 571 | 641 | 1212/1357 | 7/0 | 3 |
| V164 strict | 571 | 640 | 1211/1357 | 6/0 | 1 |
| V164 generalization | 571 | 642 | **1213/1357** | 8/0 | 11 vs public base |

The gain is stable across both folds and every held-out user: no user loses a
correct prediction. Neighboring rank thresholds retain six to eight gains
without a validation loss, so the result is not isolated to one threshold.
The generalization CSV starts from the complete locally selected V127 rather
than the public-score ablation, matching the validation protocol exactly.
