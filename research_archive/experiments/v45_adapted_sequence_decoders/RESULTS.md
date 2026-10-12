# V45 Results

Fold 0 selected repeat recovery with a maximum gap of `120`, minimum sequence
similarity `0.5`, and pooling power `1.5`.

| Metric | V32 | V45 |
|---|---:|---:|
| Fold 0 pair-aware | 551/652 | **556/652** |
| Fold 1 blind | **623/705** | 622/705 |
| Combined | 1174/1357 | **1178/1357** |

The five-row selection-fold gain does not transfer: blind Fold 1 loses one row.
The higher combined score is therefore treated as selection overfit rather than
a reliable improvement. V45 is not promoted and no test submission is built.

