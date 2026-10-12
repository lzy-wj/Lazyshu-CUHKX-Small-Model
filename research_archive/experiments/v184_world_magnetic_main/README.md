# V184 World + Magnetic Main Model

V184 appends the fifteen local magnetometer axes to the existing 85-channel
world-coordinate IMU stream.  The V23 multimodal checkpoint is loaded exactly:
all original IMU convolution weights are copied and the new magnetic-channel
weights start at zero.  This isolates the newly recovered sensor signal without
disturbing the incumbent visual, skeleton, fusion, or classifier parameters.

`standard` updates the full model with the established low backbone learning
rate.  `frozen_visual` updates only the sensor/fusion side.  Both remain below
the 100 MB quantized artifact limit.

