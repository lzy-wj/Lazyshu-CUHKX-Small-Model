# V116 Iterative Paired Adapters

V116 performs one explicit self-training loop. V114 probabilities are decoded
with the frozen pair-aware sequence model to create a stronger target teacher
than V89. The V113 paired-person adapters are then retrained from the original
V23 initialization using those refreshed hard targets. No validation labels
enter pseudo-label construction, adapter training, or inference.
