# V70 FiLM Margin Distillation

The V55 FiLM expert improves the frozen V59 sequence system from 1197/1357 to
1201/1357 when used only for domain 2 at temperature 3 and weight 0.5. The gain
comes from probability margins rather than standalone top-1 changes.

V70 distills those soft margins into a small permitted round Transformer. Its
inputs are deployable V57, V52, V24-adapted, and V24-base log probabilities,
plus modality-presence and environment/paired-source features. V55 is a
training-only teacher and is not a deployment asset.

The distillation is transductive, matching the organizer-confirmed pseudo-label
protocol already used by V24/V52/V57. Fold 0 fixes the teacher rule; Fold 1 is
evaluated with the same rule. The student must retain the teacher's decoded
gain on both folds before test deployment.

