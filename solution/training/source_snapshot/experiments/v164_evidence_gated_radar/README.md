# V164 Evidence-Gated Radar

V164 keeps V127 as the incumbent and treats the Radar batch-key decoder as a
challenger. A challenger is accepted only when its label is in the top two of
the original routed multimodal probabilities while the incumbent label is not.
The fixed rule directly tests whether structural decoding and clip evidence
agree; it does not use test labels or public-score row fitting.
