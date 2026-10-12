# V30 Hard-Pseudo Adaptation Sweep

V30 keeps the V23 student, V13 pair-aware hard teacher, model scope, data,
augmentation, TTA, and decoder fixed. Fold 0 screens one-variable changes to
adaptation epochs, learning rate, confidence threshold, and retention weight.
Only robust Fold 0 winners proceed to Fold 1.

The reference is V24 hard pseudo adaptation: Fold 0 raw `493/652`, direct
pair-aware `550/652`; two-fold direct pair-aware `1168/1357`.
