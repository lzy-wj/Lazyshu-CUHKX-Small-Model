# V136 Retained Label Experts

V136 corrects V135's catastrophic-forgetting risk. Every expert continues to
train on all source clips and all 40 labels, while a fixed class-weighted loss
emphasizes one difficult semantic family. The checkpoint endpoint is selected
only by that family's held-out accuracy. Inference will preserve total family
probability mass exactly and only replace the conditional distribution inside
the family.

The two non-overlapping specialists are:

- `documents`: typing, writing, reading documents, and turning pages
  (`17,18,21,22`).
- `devices`: calls, time checking, headphones, mobile use, TV, games, selfies,
  medicine, massage, and body temperature (`19,20,23,24,25,26,27,37,38,39`).

Both use the same V23 initialization and a fixed 4x loss weight, so the
comparison changes semantic focus rather than architecture or optimization.
