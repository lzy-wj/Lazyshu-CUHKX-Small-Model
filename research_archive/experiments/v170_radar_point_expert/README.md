# V170 Radar Point Expert

The deployed V24/V166 system uses Radar file timestamps for batch recovery but
does not encode Radar measurements for classification. The older 28-statistic
Radar branch was rejected. A richer cache already exists but was never modeled:
64 temporal frames, 24 detections per frame, and 10 point features.

V170 is a strict cross-user ablation over clips with non-empty Radar returns.
It uses a PointNet-style 1x1 CNN, temporal CNN, and compact Transformer. Fold 0
holds out users 3/8 and Fold 1 holds out users 5/7. Test labels are never read.
