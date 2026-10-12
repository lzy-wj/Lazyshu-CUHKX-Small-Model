# Results

## Full-data refits

| Recipe | Epochs | Final train accuracy | FP16 bytes | SwanLab |
| --- | ---: | ---: | ---: | --- |
| Conservative | 14 | 72.56% | 72,619,668 | `ufe8jwv0` |
| Domain-adapted | 14 | 75.69% | 72,619,125 | `lez1wpjc` |

Both models use 36,212,664 parameters. Each deployment checkpoint remains
below 100MB; the SSD preprocessing assets add about 14.1MB.

## Decision

The validation MViT variants did not beat V127: the best formal route fusion
scored 1,204/1,357 versus V127's 1,205/1,357. The full-data models therefore
serve only as independently trained evidence for test-time candidate auditing.

On V127's 11-row medium candidate, both paired MViTs agree with the challenger
at positions 333, 335, and 363. The consensus candidate builder deliberately
keeps all other V24 predictions unchanged.
