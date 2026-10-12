# V35 Adapter Seed Ensemble

V35 repeats the public-validated V24 hard-pseudo head adaptation with seeds 43
and 44. The V24 seed-42 adapter is the reference. Fold 0 selects among the two
new adapters and equal-probability seed ensembles; only the selected seed set
is reproduced on Fold 1.

Each adapter is about 0.17 MB, so even a three-forward ensemble remains far
below the 100 MB model-asset limit while keeping one shared INT8 backbone.
