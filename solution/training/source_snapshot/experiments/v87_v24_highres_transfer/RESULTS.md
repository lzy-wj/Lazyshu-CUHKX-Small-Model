# V87 Results

The frozen full-data V24 model was evaluated at 160 pixels with unchanged TTA
and pair decoding. Its pair-decoded predictions differ from the submitted V24
predictions on 13 of 405 test rows.

Replaying the exact two-epoch V24 hard-pseudo head adaptation at 160 pixels
produces a local-only candidate that differs from submitted V24 on 12 of 405
rows (`97.04%` agreement). The base model, YOLO asset, and head delta remain
about 71.1 MB in total.

This is not hidden-label evidence and has not been submitted. Promotion is
blocked until the corresponding two-fold V88 validation beats the V24
`1168/1357` local reference under the frozen acceptance rule.
