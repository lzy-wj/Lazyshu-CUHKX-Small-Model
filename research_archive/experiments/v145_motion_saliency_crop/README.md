# V145 Motion-Saliency Crop

Fine-grained residual errors are concentrated in writing, reading, turning
pages, and mobile-device use. A fixed center crop often misses the interaction
area. V145 computes a deterministic saliency center from normalized temporal
differences across Depth, Thermal, and IR, crops one square interaction region,
and resizes it to the source resolution.

The first gate is inference-only on frozen V23 checkpoints. Fold 0 compares
two preregistered crop fractions and low fusion weights. Fold 1 is launched
only for the Fold-0-selected nonzero candidate.

