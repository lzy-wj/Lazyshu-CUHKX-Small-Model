# V59 Teacher Sequence Hybrid

V59 starts from the V57 round-Transformer output, applies the established
timestamp/probability repeat recovery (`gap=150`, `similarity=0.7`, `power=1`),
then runs the frozen pair-aware decoder. It adds no model parameters.

The repeat threshold is inherited from the original repeat decoder rather
than selected from Fold 1. Test inference uses no labels or trial names.
