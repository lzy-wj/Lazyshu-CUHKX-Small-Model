# Status

The first two-epoch pilot was stopped because scale sampling consumed the same
Python RNG used by later video crops. The implementation now uses an isolated,
worker-specific RNG so all baseline augmentations retain their original random
streams. The new smoke test passed, and clean Fold 0/Fold 1 runs started on
physical GPUs 4 and 5 at 16:55 Asia/Shanghai. Both folds completed and the
preregistered gate rejected the method; no full-data refit is allowed.
