# V86 Results

## Trained spatial encoder

Using the same final epoch on both folds, increasing the V80 spatial masked
encoder input from 128 to 160 pixels changes direct pair-aware accuracy from
`1153/1357` to `1165/1357`. Test-domain-weighted pair accuracy rises from
`86.31%` to `87.79%`, with gains on both folds (`85.34% -> 87.03%` and
`87.27% -> 88.55%`). Raw top-1 falls by three rows, so the gain comes from more
sequence-coherent probabilities rather than a globally better argmax classifier.

## Frozen V23 control

With identical V23 weights and inference settings, changing only the input from
128 to 160 pixels raises direct pair-aware accuracy from `1153/1357` to
`1160/1357`. Mean test-domain-weighted pair accuracy rises from `86.34%` to
`87.32%`, again improving on both folds. This isolates higher visual resolution
as a reproducible preprocessing gain.

## Decision

Retain 160-pixel input as a genuine generalization candidate. V86 itself does
not yet exceed the stronger V24 pseudo-target reference (`1168/1357`), so it is
not a standalone replacement for the current best system.
