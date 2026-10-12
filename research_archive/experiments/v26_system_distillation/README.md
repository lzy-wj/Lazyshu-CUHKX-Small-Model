# V26 System Distillation

V24's strongest local result (`1195/1357`) uses a V14 base system and a V24
candidate system. Deploying both inference models would exceed the 100 MB
limit. This experiment converts the frozen final system decisions into
pseudo-labels, adapts only one student's heads, and then evaluates that single
student with the frozen pair-aware decoder.

The adaptation recipe is inherited unchanged from V24 hard pseudo-labeling:
two epochs, learning rate `2e-4`, retention weight `0.25`, and BF16 autocast.
No V26 hyperparameter search is performed. Fold 0 must reach at least
`545/652` under direct decoding before Fold 1 runs. Deployment promotion
requires more than the V24 direct hard-pseudo reference of `1168/1357`.

