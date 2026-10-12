# V115 Confidence-Anchored Paired Adapters

V115 changes one training principle from V113: clean counterpart-source labels
receive weight 2, while target pseudo-label weights vary continuously from 0.5
to 1 with teacher confidence. This prevents uncertain transductive labels from
overriding the only clean person-specific supervision. Architecture, data,
epochs, optimizer, retention, stream distillation, AdaBN, and TTA are unchanged.
