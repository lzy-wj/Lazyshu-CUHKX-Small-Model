# Results

| Motion crop | Raw | Pair-aware |
| --- | ---: | ---: |
| 50% side | 428/652 | 521/652 |
| 65% side | **474/652** | **542/652** |
| V127 final | - | 569/652 |

The deterministic crop visually covers the intended desk and hand regions,
and the 65% view remains reasonably recognizable by the frozen model.
Nevertheless, every preregistered route and low fusion weight reduces Fold-0
final accuracy. The experiment therefore stops before Fold 1 and before any
crop-specific fine-tuning.

