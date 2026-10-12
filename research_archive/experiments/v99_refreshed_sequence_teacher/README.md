# V99 Refreshed Sequence Teacher

V99 changes one variable in V98: the hard target labels are regenerated from
the current V89 equal multi-scale probabilities and the frozen pair decoder,
rather than inherited from the older V13 system. The refreshed teacher reaches
1170/1357 pair-aware OOF versus 1161/1357 for the old pseudo-label source.

The protected dual-stage CNN adapter, high-resolution target data, two epochs,
cross-modal soft stream distillation, retention, AdaBN, and TTA are unchanged.
