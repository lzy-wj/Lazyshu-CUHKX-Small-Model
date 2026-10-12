# Status

Completed on GPUs 4 and 5.

- Conservative refit: 14 epochs, SwanLab run `ufe8jwv0`.
- Domain-adapted refit: 14 epochs, SwanLab run `lez1wpjc`.
- Both 36.21M-parameter checkpoints were exported to FP16 at 72.62MB.
- Test probabilities and paired-template predictions are complete.
- These models are retained only as independent test evidence: both validation
  variants were materially weaker than V127 and direct fusion was rejected.
