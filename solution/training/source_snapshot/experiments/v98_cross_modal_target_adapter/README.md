# V98 Cross-Modal Target Adapter

V98 moves the protected V97 visual adapters from source-label fine-tuning to
legal transductive target adaptation. The unchanged V23 fused multimodal model
acts as the teacher. Fixed pair-aware pseudo labels supervise the final output,
while the confirmatory `cross` variant additionally distills the fused soft
distribution into each available visual stream.

The `main` and `cross` variants differ only by the cross-modal stream loss.
Both use the exact V24 two-epoch target recipe, high-resolution inputs, target
AdaBN, horizontal-flip/center-crop TTA, and a 0.6M-parameter dual-stage CNN
adapter initialized to the exact source function.
