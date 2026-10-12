# V159 Ordered Source Alignment

The source users do not execute identical labels perfectly, but their trial
blocks retain a strong chronological order across users. V127 uses a flat
source-template pool, so an attractive template can be applied to a target
round from the wrong script position.

V159 aligns target rounds monotonically to the paired source user's labeled
rounds using flexible within-round matching. Dynamic programming permits source
or target omissions; accepted matches keep the unchanged V127 unary and
transition score. This uses only source labels, timestamps, and target model
probabilities, not target labels or trial IDs.

Fold 0 selects one fixed skip penalty; Fold 1 is then evaluated once blind.
