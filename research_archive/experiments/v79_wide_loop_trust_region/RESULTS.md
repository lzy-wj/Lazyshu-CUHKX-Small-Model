# V79 Results

| Candidate | Best trained epoch | Best trained correct | Selected checkpoint |
|---|---:|---:|---:|
| Wide trust-region K1 | 9 | 482/652 | epoch 0, 487/652 |
| Wide trust-region K2 | 9 | 485/652 | epoch 0, 487/652 |

K2 gains three rows relative to the trained K1 residual under identical
parameters, which is weak evidence that one repeated pass can help fusion.
However, neither learned residual beats the untouched V23 checkpoint, and both
have substantially worse NLL. The safe epoch-0 preservation mechanism worked,
but the experiment fails the net-improvement gate and does not proceed to Fold
1 or Kaggle.

The added parameters operate after every modality has already been compressed
to 192-dimensional tokens. Increasing capacity after that bottleneck cannot
recover visual information discarded by the shared encoder.
