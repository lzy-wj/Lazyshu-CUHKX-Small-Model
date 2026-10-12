# V71 Gated High-Fidelity FiLM Distillation

V71 compresses the validated V55 probability correction into a compact
Transformer while retaining V57 as the deployment anchor. The FiLM correction
is active only in paired-source environment 0, the domain selected on Fold 0
and transferred successfully to Fold 1. All other rows are hard-gated to the
original V57 logits, preventing approximation noise from changing the frozen
sequence decoder.

The three-layer 256-dimensional Transformer is fitted transductively to
unlabeled target probabilities and serialized in FP16. V55 is training-only;
the final V57 assets plus this student remain below 100 MB.
