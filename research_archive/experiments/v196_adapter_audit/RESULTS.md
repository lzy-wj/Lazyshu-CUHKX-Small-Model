# Results

Epochs are selected on Fold 0 and transferred unchanged to Fold 1.

| Adapter | Selected epoch | Fold 0 | Fold 1 | Combined delta vs V23 |
| --- | ---: | ---: | ---: | ---: |
| Statistics, seed 42, LR 1e-3 | 3 | 480 | 565 | +1 |
| Statistics, seed 42, LR 3e-4 | 7 | 480 | 565 | +1 |
| Statistics, clean training | 4 | 480 | 564 | 0 |
| Statistics, seed 43, LR 1e-3 | 5 | 480 | 565 | +1 |
| Device attention, seed 42 | 4 | 481 | 560 | -3 |
| Device attention, seed 43 | 6 | 480 | 560 | -4 |

The simple statistical residual contains a reproducible but one-row signal.
It does not alter the stronger V166 sequence system under conservative routing.
The higher-capacity device-attention residual overfits Fold 0 and is rejected.
