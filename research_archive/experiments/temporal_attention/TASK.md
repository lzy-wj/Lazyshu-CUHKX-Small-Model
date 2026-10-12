# Temporal Attention Ablation

Change only the multimodal fusion operator from the existing Transformer token fusion to relative-time-aware cross-modal attention. Preserve encoders, token dimensions, preprocessing, augmentations and training settings. Keep parameter count comparable and report fold-0 results against the frozen reference.
