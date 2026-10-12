# V111 Robust Adapter Consensus

V104 uses an arithmetic mean of diverse pseudo-adapted models. V111 tests
standard robust aggregation rules that can suppress one miscalibrated adapter
without fitting blend weights: normalized geometric pooling, coordinate-wise
median pooling, and strict top-1 majority acceptance.

All routes, component models, and pair decoding remain fixed. Fold 0 selects
one aggregation rule by target-domain-weighted pair accuracy, then Fold 1 is
evaluated once with that frozen rule. This is a discrete robustness test, not a
coefficient search.
