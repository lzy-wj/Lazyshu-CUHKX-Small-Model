# V195 Official Exact IMU

This experiment reproduces the September 2026 official CUHK-X IMU baseline as
closely as possible under the fixed local subject-held-out folds. It uses raw
accelerometer, gyroscope, and magnetometer values from five devices, Fourier
resampling to 50 steps, three 1D CNN layers, and a two-layer Transformer.

The experiment is isolated from test labels and does not modify any deployment
candidate unless both the standalone and fixed-transfer fusion gates pass.
