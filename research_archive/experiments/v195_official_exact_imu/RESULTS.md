# Results

The exact official preprocessing and CNN-Transformer recipe strongly overfits
the source subjects and does not generalize to the fixed held-out users.

| Fold | Best epoch | All rows | IMU-present rows |
| --- | ---: | ---: | ---: |
| 0 | 41 | 149/652 (22.85%) | 149/622 (23.95%) |
| 1 | 7 | 174/705 (24.68%) | 165/603 (27.36%) |

Training accuracy approached 99%. This confirms that the official baseline's
cross-trial recipe is not a competitive cross-subject component here. It is
rejected without fusion or test prediction.
