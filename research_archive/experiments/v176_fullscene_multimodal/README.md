# V176 Full-Scene Multimodal

The V24 family encodes YOLO person crops, while the remaining errors are
concentrated in object-dependent actions such as reading, turning pages,
writing, medicine, phones, and tableware. V176 keeps the full scene.

Four relative-time frames from each of Depth_Color, Thermal, and IR pass
through a shared ImageNet ResNet18. Their tokens join Skeleton and IMU tokens
in a two-layer Transformer. The model is evaluated on both paired-identity-
isolated folds and serialized as INT8, comfortably inside the remaining model
budget if it proves complementary.
