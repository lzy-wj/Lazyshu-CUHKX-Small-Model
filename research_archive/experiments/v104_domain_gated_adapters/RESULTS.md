# V104 Results

| Candidate | Pair-aware | Target-domain weighted pair | Fold deltas vs V89 |
|---|---:|---:|---:|
| V89 | 1170/1357 | 88.81% | - |
| Conservative domain-1 gate | 1173/1357 | 89.12% | +1 / +2 |
| Full domain-1/2 gate | **1175/1357** | **89.25%** | **+2 / +3** |

The conservative route changes only the unmatched high-environment domain.
The full route additionally uses the protected source adapter in the paired
low-environment domain. Both improvements are present on both folds; the full
route is the new local best.

Full-data old-teacher and refreshed-teacher target adapters were trained for
the fixed two epochs on 355 and 400 selected test pseudo labels. Their SwanLab
runs are `xtfrtz86` and `egfpd8xd`. INT8 deployment changes one component
top-1 prediction in each adapter relative to its FP32-source training graph;
the final candidate is built only from the reproduced INT8 predictions.

The complete deployable asset set is `75,355,331` bytes: one V24 INT8 base,
YOLO11n, two V89 head deltas, the V103 protected adapter, and both V104 adapted
states. The full and conservative pair-aware test candidates remain marked
`do_not_submit` pending the final model comparison.
