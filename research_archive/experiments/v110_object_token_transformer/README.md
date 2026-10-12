# V110 Object-Token Transformer

The deployed YOLO asset was previously used only to obtain a person crop. V110
reuses the same detector without adding model assets and retains its complete
COCO output: class confidence, box center, width, height, and count for 80
classes over eight phases of Depth-Color, Thermal, and IR.

A small Transformer treats the 24 phase-modality observations as tokens. It is
trained from labels only, with inverse square-root class weighting and
detection dropout. This supplies explicit object and scene evidence for the
fine-grained hand-object classes that remain outside V104's top-k predictions.

Feature extraction is label-independent and shared by both user-isolated
folds. The primary fusion is a fixed equal average with V104 only when either
model predicts a semantically defined object-interaction action (labels 6-27
or 37-39). No fusion-weight search is used.
