# V72 Modality-BN FiLM

V72 tests whether the visual bottleneck is normalization interference between
Depth, Thermal, and IR. It shares every R(2+1)D convolution across modalities,
but gives each modality independent BatchNorm state and the same 6,144-parameter
FiLM adapters validated by V55. Sensor encoders, Transformer fusion, data,
augmentation, optimizer, seed, and Fold-0 protocol remain unchanged.

The extra BatchNorm parameters and buffers are negligible relative to the
65 MB quantized checkpoint, so the deployment remains below 100 MB.
