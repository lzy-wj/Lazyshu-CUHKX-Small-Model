# V101 Native 224-Pixel Multi-Scale

V101 extends the only visual preprocessing change that improved both held-out
folds: native-resolution person crops. It rebuilds all three visual streams
directly from original frames at 224 pixels using the exact frozen YOLO windows
from the 160-pixel cache. It does not resize an existing low-resolution cache.

The first confirmatory test evaluates unchanged V23 weights at 224 pixels. If
the direct scale audit is positive on both target-weighted folds, the exact V88
two-epoch pseudo adaptation and fixed equal-scale ensemble are repeated.
