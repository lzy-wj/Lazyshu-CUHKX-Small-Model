# V150 Group-DRO User Generalization

V150 targets the dominant cross-subject failure directly. Instead of minimizing
the average clip loss, it maintains adversarial weights over training users and
minimizes the resulting worst-user risk. Mixup mass is assigned to both source
users, so the robust objective remains well-defined under the V23 recipe.

The confirmatory run warm-starts V23, keeps its architecture and augmentations
fixed, and changes only the task-risk aggregation. A single conservative
`eta=0.01` is used. Fold 1 is allowed only after a Fold-0 system-level gain.
