# V172 Thermal Frame ResNet

The official CUHK-X benchmark reports Thermal as the strongest standalone
sensor and trains ImageNet ResNets on 224px frames. Our previous Thermal branch
was a small scratch video encoder and is not comparable.

V172 reproduces the official frame-CNN idea with strict subject-isolated folds,
random frame sampling during training, and 12-frame clip aggregation during
validation. It uses ResNet34 and stores INT8 weights so a successful expert can
remain under the total 100 MB deployment budget beside the quantized V24 model.

The same controlled implementation is also used for IR and Depth_Color after
the Thermal run, changing only the source modality.
