# V128 MViT Visual-Domain Adaptation

V126 remains strongly underfit after eight epochs despite consistent validation
gains: training accuracy is only 66-69%, while the Kinetics visual backbone
receives at most 1.5e-5 learning rate. V128 keeps the full five-modal MViT
architecture and augmentation, but raises the visual-backbone factor from 0.05
to 0.15 and reduces mixup from 0.2 to 0.1. This tests whether adapting the RGB
pretraining to false-color Depth/Thermal/IR is the limiting factor.

