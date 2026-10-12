# V197 Device-Attention IMU Adapter

V197 preserves the complete V23 multimodal path and adds a zero-initialized
logit residual. Each of the five official IMU devices is encoded as a separate
statistical token. The frozen fused multimodal representation queries those
tokens through multi-head attention, allowing class-conditional sensor use
without disturbing the inherited prediction at initialization.

Only the new 249k-parameter branch is optimized. Epoch selection uses Fold 0
and transfers unchanged to Fold 1.
