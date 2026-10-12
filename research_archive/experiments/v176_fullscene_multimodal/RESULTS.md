# Results

The full-scene five-modality model reached `312/652` on Fold 0 and `341/705`
on Fold 1. It fitted the source users to roughly 98%, confirming that the
remaining gap is cross-user generalization rather than insufficient capacity.

Against V166, the raw model has only four and five oracle corrections. Passing
its evidence through the frozen sequence decoder raises those upper bounds to
six corrections per fold, but every fixed low-weight blend that preserves Fold
0 loses at least two rows on Fold 1. The Fold-0 optimum ties V166; it does not
improve it.

Decision: reject full-scene ResNet fusion as a deployment component. Object
context alone does not provide sufficiently independent errors.
