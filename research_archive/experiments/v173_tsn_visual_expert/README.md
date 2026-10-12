# V173 TSN Visual Expert

V172 applies a clip label independently to random frames and heavily overfits.
V173 uses Temporal Segment Network supervision: sample one frame from each of
four temporal segments, average logits, then compute one clip-level loss. The
ImageNet ResNet34 backbone also uses one tenth of the classifier learning rate.
This directly targets cross-subject generalization while retaining full-frame
object and scene context.
