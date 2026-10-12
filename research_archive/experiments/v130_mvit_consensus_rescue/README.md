# V130 MViT Consensus Rescue

Soft MViT fusion dilutes the stronger sequence system. V130 instead treats the
two independently trained MViTs as a conservative verifier. A V127 prediction
may be replaced only when the conservative and strongly domain-adapted MViTs
agree on the alternative and both exceed the same confidence threshold.
Fold 0 selects a coarse threshold and supervision domain; the fixed rule is
then evaluated once on Fold 1. A no-change control is eligible.

