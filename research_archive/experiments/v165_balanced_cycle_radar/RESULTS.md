# Results

| System | Fold 0 | Fold 1 | Combined | Gains/losses |
| --- | ---: | ---: | ---: | ---: |
| V127 | 569 | 636 | 1205/1357 | - |
| Balanced cycles | 573 | 644 | 1217/1357 | 12/0 |
| Cycles + V164 | **574** | **647** | **1221/1357** | **16/0** |

The union improves every non-neutral held-out user and never reduces a user's
accuracy. It accepts 20 validation changes: 16 corrections and four
wrong-to-wrong changes, with no newly incorrect row.
