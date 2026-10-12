# V152 Balanced Softmax

CUHK-X is strongly long-tailed: the training labels range from 12 to 365 clips.
The official class-balanced resampling result motivates treating this as a real
representation problem, but V146 showed that replacement sampling disrupts the
warm V23 optimization path.

V152 instead uses canonical Balanced Softmax. Training cross-entropy receives
the log training-class prior while inference remains unchanged. This corrects
long-tail decision boundaries without duplicated clips, extra parameters, or
an inference-time prior. Temperature 1 is the method default; no sweep is used.

Fold 1 is allowed only after a Fold-0 V127-aware gain.
