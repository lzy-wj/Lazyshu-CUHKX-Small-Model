# Final Submission Plan — 2026-09-15

## Submitted results

- `v164_evidence_gated_radar.csv`: `0.91044` (submission `56243275`).
- `v166_repeat2.csv`: `0.88557` (submission `56243279`).
- Both are below the frozen `v127_without_0531.csv` score of `0.92039`.
- Stop after these two submissions; three daily opportunities remain unused.

## Frozen candidates

- Public incumbent: `experiments/v160_radar_batch_decoder/candidates/v127_without_0531.csv`
  - observed public score: `0.92039` (`185/201`)
  - SHA256: `76210e13f1f63a3d63c356f2462aa09917a6d9c0c919a0e8025098ab4408acb0`
- Generalization candidate: `experiments/v166_repeated_transition_consensus/candidates/v166_repeat2.csv`
  - strict local validation: `1226/1357`
  - 21 gains and zero losses versus V127 across both folds
  - all 60 partial-script stress views introduce zero losses
- Pure local-base variant: `experiments/v166_repeated_transition_consensus/candidates/v166_repeat2_generalization.csv`
  - use only as an audit artifact because it reintroduces four source-20
    changes removed by public evidence

## Five-slot decision

Do not submit automatically. After explicit user approval, use the daily slots
sequentially and inspect each completed score before continuing:

1. Submit V164 strict (one evidence-gated change).
2. Submit V164 conservative (adds one two-row group).
3. Submit V164 main (adds one four-row group).
4. Submit V166 repeat-2 from the public incumbent.
5. Compose and submit the best additive combination implied by slots 1–4.

The already submitted `v127_without_0531.csv` remains a valid fallback and can
be selected as one final submission without consuming another slot. The second
final selection should favor V166 only if the new public evidence is neutral or
positive; otherwise retain V166 as the private/generalization hedge only when
competition final-selection rules permit choosing the slot-4 submission.

## Current evidence

- V176 full-scene multimodal: `312/652` and `341/705`; rejected.
- Perfect validation repeat grouping reaches only `543/652` and `624/705`
  from the frozen evidence, below V166.
- Universal unmatched-user script templates make no stable correction.
- Count-modulo-three repair loses on Fold 0 and is rejected.
- No Kaggle submission session or process is active.

## Final research update

- The official magnetometer omission was corrected and audited. It yields only
  `+1` stable raw row through the safest adapter and does not improve V166.
- Exact official IMU CNN-Transformer reproduction reaches 22.85%/24.68% on the
  two subject-held-out folds and is rejected.
- Five-device attention residuals score `-3/-4` under Fold0-to-Fold1 transfer.
- Do not spend the remaining three slots. Select `v127_without_0531.csv` as the
  primary final entry and submitted V164 as the moderate private hedge. Full
  V127 is safer but too similar to the primary; V166 is rejected by its
  seven-row public loss.
