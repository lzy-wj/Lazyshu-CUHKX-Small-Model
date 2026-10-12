# V25 Results

Fold-0 screening completed on 2026-09-10. Fold 1 was not trained or inspected.

| Metric | V23 reference | V25 |
|---|---:|---:|
| Raw correct | 487/652 | 486/652 |
| Domain pair-aware / system gate | 566/652 | 559/652 |

The soft target preserved raw accuracy but weakened the downstream decoder and
second-stage system. The preregistered Fold-0 gate rejected the candidate, so it
will not be adapted on the test set or submitted to Kaggle.

