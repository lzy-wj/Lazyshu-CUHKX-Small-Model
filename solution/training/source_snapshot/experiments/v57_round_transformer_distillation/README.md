# V57 Round Transformer Distillation

V57 compresses the remaining sequence behavior of the frozen system teacher
into a small permitted Transformer. It consumes only normalized log
probabilities from the deployable V52, V24 adapted, and V24 base predictions.
Samples are grouped into acquisition rounds using timestamps; no sample IDs,
labels, or trial names are model inputs.

Hyperparameters are fixed on Fold 0 and evaluated once on Fold 1 with the same
target-domain pseudo-distillation protocol as V52. The test teacher is used
only to create training targets; it is not part of the final model assets.
