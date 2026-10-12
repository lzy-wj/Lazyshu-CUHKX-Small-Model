# V89 Pseudo-Adapted Multi-Scale Inference

V89 averages the probabilities from the same V23 base model after the fixed V24
pseudo-target adaptation at 128 and 160 pixels. The arithmetic mean is fixed at
equal weight; no blending coefficient is searched.

This tests scale marginalization through repeated forward passes. It adds only a
second 81,120-parameter head delta, keeps the deployable assets below 100 MB,
and uses the unchanged pair-aware decoder.
