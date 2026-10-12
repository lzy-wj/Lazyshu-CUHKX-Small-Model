# Status

- Repeated after freezing the finalized cache.
- The frozen-visual variant completes cleanly and reaches `480/652` and
  `566/705`, two raw rows above its V23 initialization.
- Fold-0 routing selects the unchanged unmatched path; every decision-changing
  paired blend loses at least one Fold-0 row. The stronger V166 system is not
  improved, so V184 is not promoted.
- Full-model training is stopped after three epochs because Fold 0 remains at
  or below its source while the same transferred epochs regress Fold 1.
