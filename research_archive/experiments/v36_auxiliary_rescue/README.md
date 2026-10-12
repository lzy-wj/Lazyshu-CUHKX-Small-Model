# V36 Auxiliary Rescue

V36 reuses the skeleton auxiliary head already trained inside the V23/V24
multimodal model. It adds no model asset and no extra encoder. Auxiliary
predictions are generated with the same target-domain AdaBN pass used by hard
pseudo adaptation, then blended only where skeleton is available.

Fold 0 selects a compact temperature, weight, and scope grid. Fold 1 is
evaluated once with the selected setting.
