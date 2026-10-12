# Results

| Minimum repeats | Fold 0 | Fold 1 | Combined | Gains/losses |
| ---: | ---: | ---: | ---: | ---: |
| V127 | 569 | 636 | 1205/1357 | - |
| 3 | 574 | 650 | 1224/1357 | 19/0 |
| 2 | **574** | **652** | **1226/1357** | **21/0** |

Both repeat thresholds improve both folds and every held-out user is
non-decreasing. Threshold two adds five correct rows over V165 without a new
validation error.

Across 60 user-stratified partial-script stress views, V166 never introduces a
new error. At 50%, 70%, and 85% row retention, worst-case gains are respectively
3, 8, and 15 rows.
