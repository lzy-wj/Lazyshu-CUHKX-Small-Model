# V24 Full-Data Results

| Candidate | Local direct pair-aware | Kaggle public | Submission |
|---|---:|---:|---:|
| Five-source INT8 soup | 1153/1357 CV reference | 0.87064 | 56140362 |
| V13 hard-pseudo adaptation | **1168/1357** | **0.91044** | 56140412 |
| V52 domain residual compression | 1193/1357 | pending | pending |
| V57 round-Transformer compression | **1195/1357** | pending | pending |
| V59 teacher-sequence hybrid | **1197/1357** | pending | pending |

All five source models were refit on all `3036` training clips. The deployed
INT8 checkpoint plus YOLO asset is `70,941,076` bytes. Hard pseudo adaptation
adds a `167,571`-byte head delta, for a total of `71,108,647` bytes.

The unadapted full-data soup underperforms V13 publicly, while adapting its
heads toward the V13 pair-aware teacher creates the first result above `0.90`.
This establishes single-teacher hard pseudo adaptation as the primary research
direction.

V52 and V57 subsequently compress the frozen multi-model system teacher into
small residual modules. V57 reproduces every local and test teacher decision
with total deployable assets of `71,324,325` bytes. Its CSV differs from V52 in
five rows and is the first submission candidate after the daily quota resets.
V59 adds the frozen repeat/sequence decoder, reaches `1197/1357`, and replaces
V57 as the first candidate scheduled after quota reset.
