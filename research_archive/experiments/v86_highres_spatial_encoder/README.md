# V86 High-Resolution Spatial Encoder

V86 changes one input variable in the near-winning V80 masked spatial encoder:
the same fixed person/context windows and 32 sampled frames are decoded at
`160x160` instead of `128x128`.

This tests a specific interaction rather than generic resolution scaling. V80
preserves an `8x2x2` layer-3 grid for Transformer fusion, while the ordinary
high-resolution branch pooled space too early. The V23 parameter-soup base
remains frozen, the V80 architecture/losses remain unchanged, and both folds
use the original V80-selected six-epoch schedule.

The inference graph remains about 76.2M parameters and its INT8 package remains
under 100MB because input resolution does not change parameter count.
