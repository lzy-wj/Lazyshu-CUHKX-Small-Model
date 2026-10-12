# V182 Magnetic IMU

The original local IMU cache omitted all magnetometer columns.  The official
CUHK-X IMU baseline uses acceleration, gyroscope, and magnetometer from all five
wearable devices.  This experiment restores that signal without using test
labels or identity metadata.

- `official`: 45 physical channels plus five device-presence channels.
- `magnetic`: the same magnetometer channels added to the existing Euler and
  quaternion representation, for 95 physical plus five presence channels.

The representation-only ExtraTrees diagnostic improved on both held-out folds:
`222/622` and `277/603` for `official`, versus `205/622` and `260/603` for the
old raw representation.  Formal models use the existing random-init
CNN+BiLSTM protocol and the same Fold-0-selection/Fold-1-transfer gate as V181.

