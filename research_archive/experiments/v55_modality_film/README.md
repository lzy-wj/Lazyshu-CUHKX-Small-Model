# V55 Modality-Conditioned Visual Adapter

V55 is a controlled Fold-0 architecture ablation against V18. It keeps the
five input modalities, joint visual batching, optimizer, augmentation, seed,
epoch budget, CNN/RNN/Transformer fusion, and validation split unchanged.

The only change is a zero-initialized per-modality channel-wise affine adapter
after the shared 3D-CNN stem and each residual stage. This retains shared batch
normalization while allowing Depth, Thermal, and IR to specialize with only
6,144 additional parameters. The INT8 deployment target remains below 100 MB.
The shared 3D-CNN keeps the baseline `0.1x` backbone learning-rate factor,
while the new zero-initialized adapters train at the full head learning rate.

Fold 1 runs only if Fold 0 improves the V18 endpoint (`475/652`), gains at
least two rows under V52 integration, or produces a positive pair-aware gain
at two neighboring blend weights while reducing V52 NLL by at least 10%.
