# V108 Center-Detail Visual Expert

V104's residual errors concentrate in fine-grained hand-object actions, and
the correct class falls outside its top five for a nontrivial fraction of the
unmatched target domain. V108 therefore changes the visual evidence rather
than the decoder or model capacity.

The input is a deterministic center `128 x 128` crop from the existing
`192 x 192` label-independent YOLO person window. This is an interpolation-free
1.5x detail view centered on the torso, hands, and nearby table region. The
multimodal model still receives the original Skeleton and IMU streams, so the
detail view specializes visual evidence without discarding body dynamics.

The exact V23 warm masked-modality-distillation recipe is reused from the same
fold's V18 initialization. The only training change is the derived visual
cache. Validation will report the detail expert standalone, a fixed equal
global ensemble, and a semantic object-action gate fixed from action names
before results are inspected.
