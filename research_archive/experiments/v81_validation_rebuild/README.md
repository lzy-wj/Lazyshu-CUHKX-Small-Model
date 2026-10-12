# V81 Validation Rebuild

This audit checks whether the original eight-user OOF score predicts the public
leaderboard. It compares nine submitted systems, reconstructs public scores as
integer correct counts out of 201, and reweights OOF domains to match the 405
unlabeled test clips.

Run `python experiments/v81_validation_rebuild/audit.py` after regenerating the
V23 and V24 direct pair-aware reports. The proposed protocol is frozen in
`protocol.json`; V24 test predictions are a guardrail rather than pseudo ground
truth.
