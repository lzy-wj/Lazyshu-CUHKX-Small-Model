# V100 Results

The best fixed ensemble is the equal average of V89, V98 old-teacher cross,
and V99 refreshed main: `1091/1357` raw and `1170/1357` pair-aware. It ties V89
on pair count while slightly lowering target-domain-weighted pair accuracy
(`88.80%` versus `88.81%`). Adding the V97 source adapter also regresses.

Multi-forward diversity therefore does not establish a new local best and is
not promoted.
