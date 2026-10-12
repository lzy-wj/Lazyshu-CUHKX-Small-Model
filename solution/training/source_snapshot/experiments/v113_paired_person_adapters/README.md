# V113 Paired-Person Visual Adapters

The official split contains cross-environment recordings of the same people:
validation analogues are `8 <-> 18` and `7 <-> 17`, while test uses
`10 <-> 20`, `11 <-> 21`, and `26 <-> 16`. V113 moves this information from
the decoder into parameter-efficient visual adaptation.

For each target person, a separate V97 spatiotemporal visual adapter is trained
on the paired source person's true labels and the target person's refreshed
sequence pseudo labels. The R(2+1)D, Skeleton, IMU, and fusion backbone remain
frozen, with teacher retention and per-stream distillation. AdaBN is estimated
only on the target person.

The validation checkpoint excludes both people before adaptation. Source labels
are intentionally introduced only in the target-specific adapter, exactly
matching the legal test condition where the source person belongs to training.
No target ground-truth labels or identity-dependent tuning are used.
