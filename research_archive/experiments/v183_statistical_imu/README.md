# V183 Statistical IMU

V182 confirms that sequence CNN+BiLSTM encoders overfit wearable style.  V183
replaces only the IMU encoder with a deterministic temporal statistics pyramid
(global and 2/4-segment moments) followed by a small MLP.  This preserves the
cross-user signal found by the ExtraTrees diagnostic while remaining a valid
CNN/RNN/Transformer-family neural submission component.

Variants test official `acc+gyro+mag`, the expanded magnetic representation,
and independent world/official streams with late concatenation.

