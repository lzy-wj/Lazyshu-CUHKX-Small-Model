# V32 Domain Adaptation Gate

V32 blends the unadapted and V13-hard-adapted probabilities from the same V24
checkpoint. Fold 0 selects one adapted weight for each of four domains formed
by environment and paired-script status. Those four weights transfer unchanged
to Fold 1. Ties prefer the already public-validated adapted prediction.

Deployment needs one INT8 checkpoint, one small head delta, and two forwards;
it does not add a second backbone asset.
