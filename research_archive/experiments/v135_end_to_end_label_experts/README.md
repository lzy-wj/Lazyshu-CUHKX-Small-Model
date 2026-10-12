# V135 End-to-End Label Experts

V135 tests hierarchical class-family specialization without changing the
validated global model. Same-fold V23 R(2+1)D multimodal checkpoints are
fine-tuned end to end on fixed semantic families, with a low visual-backbone
learning rate. At inference the expert can only redistribute the incumbent
probability mass inside its family, so unrelated classes are exactly retained.

Two families are preregistered:

- `desk`: labels 17-27, covering writing, documents, pages, phones, and devices.
- `interaction`: hand-object classes fixed from action semantics before training.

The experiment differs from V95 because V95 froze the entire encoder and
trained residual Transformer heads. V135 adapts the visual and multimodal
representation itself while retaining the source initialization.
