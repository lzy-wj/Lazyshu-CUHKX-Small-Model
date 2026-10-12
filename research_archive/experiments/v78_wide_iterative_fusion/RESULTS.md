# V78 Results

| Candidate | Correct | NLL | Parameters | INT8 + YOLO |
|---|---:|---:|---:|---:|
| Protected epoch-0 V23 | 487/652 | 1.17912 | 64,953,671 | about 71 MB |
| Wide K1 | 473/652 | 1.27601 | 89,538,928 | 95,482,796 bytes |
| Wide K2 | 466/652 | 1.38291 | 89,538,928 | 95,482,796 bytes |

Unconstrained end-to-end fine-tuning destroys the strong initialization. The
equal-parameter K2 model is seven rows worse than K1, so neither additional
fusion capacity nor repeated fusion passes the Fold-0 gate.

