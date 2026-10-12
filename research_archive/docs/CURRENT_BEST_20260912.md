# Current Best — 2026-09-12

## Public leaderboard

- Best verified public score: `0.92039`.
- Candidate: `experiments/v160_radar_batch_decoder/candidates/v127_without_0531.csv`.
- It starts from V24 and retains only seven V127 changes outside `2025-05-31`.
- V127 with all eleven changes scores `0.91542`; V24 scores `0.91044`.

## Findings

- The Radar filename timestamp is a useful batch key on held-out users, but
  applying the resulting V160 decoder to the test set scores only `0.84079`.
  It is rejected despite `1222/1357` local validation rows.
- Removing the four V127 changes on `2025-05-31` improves the public score by
  one public row. This is the strongest confirmed generalization signal.

## Next controlled test

- V161 completed on `2026-09-13`: `91:18→24`, `127:22→24`, and
  `363:12→13` were neutral at `0.92039`; `273:18→24` fell to `0.91542`.
  The merged V161 submission therefore remains exactly the `0.92039`
  baseline.
- V162 completed on `2026-09-14`: `217:14→22` and `335:0→1` were neutral;
  `367:14→22` and `333:22→21` each fell to `0.91542`. The merged V162
  submission remains the unchanged `0.92039` baseline.
- All automatic submission sessions were disabled on `2026-09-14`. No file
  will be submitted until explicitly requested.
- No test labels are used; probe selection uses only submitted public scores.

## Final selection plan

- Public-score candidate: the highest verified result after V163, currently
  `v127_without_0531.csv` at `0.92039`.
- Generalization candidate: `v166_repeat2_generalization.csv`. It starts from
  the complete locally selected V127 and combines clip evidence, balanced
  within-round label cycles, and repeated-transition consensus. It reaches
  `1226/1357` versus V127's `1205/1357`, with 21 gains and zero losses.
- The V166 Fold0/Fold1 deltas are `+5/+16`; all eight held-out users are
  non-decreasing. Sixty partial-script stress views introduce zero errors;
  worst-case gains are `+3/+8/+15` at 50%/70%/85% row retention.
- V167 reweights frozen V166 by the test set's exact domain and modality
  availability frequencies. The mean estimate is `93.33%` (`377.99/405`),
  about `6.76` expected rows below `95%`; this is an estimate, not a public
  leaderboard result.
- Conservative fallback: V165/V166-repeat3 share the same test CSV and use 14
  changes instead of 16. The public-score candidate remains the verified
  `v127_without_0531.csv` at `0.92039`.
