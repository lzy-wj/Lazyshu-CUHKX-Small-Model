# V33 Domain Gate with Calibrated Decoder

V33 composes two independently selected changes:

- V32 uses 50% adapted probabilities only for unmatched environment-1 rows;
- V31 uses gap penalty `1.0` and maximum edits `6` in the pair-aware decoder.

Both components select settings on Fold 0 and transfer unchanged to Fold 1.
The deployed model assets are identical to the V24 hard-pseudo candidate.
