# Results

- Fold 0 Radar-only accuracy: `44/311` (`14.15%`).
- Frozen V166 is correct on `286/311` of the same Radar-available rows.
- Radar adds zero oracle corrections: every Radar-correct row is already
  correct in V166, while every confidence-gated replacement loses rows.
- Decision: reject the point-cloud expert and do not spend a Fold 1 run.
