# V142 Cross-Environment Metric

V142 learns only a small residual projection on frozen V23 embeddings. Training
episodes use known cross-environment person pairs `(6,16)`, `(7,17)`, `(8,18)`,
and `(9,19)`: labeled clips from one side become class prototypes and clips
from the other side are queries. Both directions are optimized.

For each fold, both members of its held-out validation pair are excluded from
metric training. The learned metric therefore transfers to unseen people and
cannot memorize validation labels. Fold 0 selects the feature family and
fusion strength; Fold 1 remains blind.

