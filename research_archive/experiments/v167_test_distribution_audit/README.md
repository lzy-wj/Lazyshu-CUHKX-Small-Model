# V167 Test-Distribution Audit

V167 does not train or alter predictions. It reweights frozen V166 validation
results by the test set's domain and exact modality-availability distribution.
This removes the known mismatch caused by validation-only sparse-modality rows
while preserving subject-disjoint folds.

Run:

```bash
$HOME/miniconda3/envs/Train/bin/python experiments/v167_test_distribution_audit/evaluate.py
```
