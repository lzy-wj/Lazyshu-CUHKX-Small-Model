# V54 Repeat Recovery on V52

V52 probabilities make repeated scripted takes more separable than earlier
systems. V54 re-runs the existing label-free repeat recovery search on Fold 0,
then applies the selected acquisition-gap and probability-similarity thresholds
once to Fold 1.

The selected settings are:

- maximum inter-round gap: `150` seconds;
- minimum Hellinger-affinity similarity: `0.7`;
- arithmetic probability pooling (`power=1.0`).

No test `trial` metadata or labels are used. Test trial names are anonymized, so
the stronger local oracle based on known training prefixes is deliberately not
deployed.

