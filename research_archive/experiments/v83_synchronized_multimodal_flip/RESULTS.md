# V83 Results

## Decision

Reject. Synchronized multimodal horizontal flipping does not improve the
V23 warm masked-distillation control.

| Variant | Raw correct | Mean test-domain weighted raw | Pair-aware correct | Mean test-domain weighted pair |
| --- | ---: | ---: | ---: | ---: |
| V23 control | 1044 / 1357 | 76.56% | 1154 / 1357 | 86.81% |
| V83 synchronized | 1041 / 1357 | 76.39% | 1139 / 1357 | 85.87% |
| 50/50 probability ensemble | 1040 / 1357 | 76.52% | 1147 / 1357 | 86.07% |

V83 raises Fold-0 raw test-domain-weighted accuracy from `72.81%` to `73.19%`,
but lowers Fold 1 from `80.31%` to `79.59%`. After pair-aware decoding it loses
15 samples overall. The result does not satisfy the frozen both-fold promotion
gate in `outputs/evaluation.json`.
