# Results

| Family | Fold 0 V127 / expert | Fold 1 V127 / expert |
| --- | ---: | ---: |
| Documents | 43/73 / 37/73 | 31/48 / 30/48 |
| Devices | 93/120 / 81/120 | 67/88 / 63/88 |

The class-weighted experts improve direct family top-1 relative to their
single V23 initialization, but V127's routed ensemble is stronger. Every
tested conditional-mass blend reduces Fold-0 system accuracy. The selected
zero-weight control remains 1,205/1,357.

V135 and V136 jointly reject both subset-only and retained-all-class label
experts as a path to a stronger teacher, so no compression stage is run.
