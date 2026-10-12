# V186 Official IMU Transformer

This experiment reproduces the official CUHK-X IMU design more closely: three
temporal CNN layers, positional encoding, and two Transformer encoder layers on
the restored acceleration/gyroscope/magnetometer channels.  A learned flatten
projection retains temporal order before classification.  Both folds use
random initialization and no external weights.

