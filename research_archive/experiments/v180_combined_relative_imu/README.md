# V180 Combined Relative IMU

The deployed R(2+1)D system uses world-frame IMU features. V180 appends a
second chest-relative representation that removes absolute sensor orientation
while retaining the original 85 channels unchanged. The new input weights are
zero-initialized, so warm initialization exactly preserves the incumbent IMU
encoder before training.

Two controlled variants use the V23 recipe on both subject-held-out folds:

- `standard`: fine-tune the complete multimodal model.
- `frozen_visual`: freeze the visual encoder and update structured encoders and
  fusion only.

Both variants keep the architecture, augmentation, optimizer, and decoder
fixed apart from the IMU representation and the declared visual-freeze
ablation.
