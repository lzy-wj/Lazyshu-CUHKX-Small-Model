# V185 Statistical-Residual Main Model

This is the full multimodal follow-up to V183.  It keeps the V23 IMU
CNN+BiLSTM stream and adds the temporal statistics pyramid as a gated residual.
The scalar gate starts at zero, so the initialized network exactly reproduces
the source model before optimization.  The temporal stream receives the old
world-coordinate channels plus the official 50-channel IMU representation,
while the statistics residual sees only the official channels.

The standard variant uses the established full-model schedule.  The
frozen-visual variant uses a higher sensor/fusion learning rate because the
visual backbone is unchanged and the new statistics branch starts from random
weights.
