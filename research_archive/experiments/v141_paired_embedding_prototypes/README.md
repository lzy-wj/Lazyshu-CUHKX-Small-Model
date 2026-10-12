# V141 Paired Embedding Prototypes

V141 tests a supervision mechanism that is different from V113 adaptation.
For a validation target user, the paired user's labeled clips form class
prototypes in the frozen V23 multimodal embedding space. No target labels or
target gradients are used. This mirrors the test pairs whose opposite-domain
recordings are present in the labeled training set.

The extraction records the fused Transformer class token, each visual stream,
Skeleton, and IMU embeddings for center and horizontal-flip views. Fold 0
selects one representation, prototype rule, and a coarse evidence strength;
that choice is transferred unchanged to Fold 1.

