# V57 Results

| Metric | V52 | V57 direct | V57 plus old decoder |
|---|---:|---:|---:|
| Fold 0 | 568/652 | 567/652 | 568/652 |
| Fold 1 | 625/705 | 628/705 | 626/705 |
| Combined | 1193/1357 | **1195/1357** | 1194/1357 |

The 92,392-parameter round Transformer reproduces all `1357/1357` local
system-teacher decisions. The teacher is already sequence-decoded, so running
the older pair-aware decoder again loses one net row; V57 therefore deploys
its direct output.

On test, the serialized 192,573-byte Transformer reproduces all `405/405`
teacher decisions and changes five final labels relative to V52. The complete
deployable asset set is `71,324,325` bytes, leaving `28,675,675` bytes below
the decimal 100 MB limit. The teacher remains training-only and is not a
deployment asset.

Candidate submission:
`experiments/v24_full_refit/outputs/submissions/v57_round_transformer_distilled.csv`

SHA256: `837391d7e50f9002e8ba9c5300d56d8a4eaff5a2b5c25685ac1100251ea9dd67`

A clean second deployment reproduced both the model bundle SHA256
`bb75157341ec056fe2b3abc934280795cc63072dc69111b47e92bfc1e4aa9667`
and the submission SHA256 exactly.
