# V153 Hierarchical Decoder

V153 is the inference-side counterpart to V151. For each clip, fine-class
probabilities are multiplied once by their official category probability and
renormalized before the unchanged V127 sequence decoders run. This is a fixed
category-evidence product, adds no fitted model, and introduces no label prior.

Fold 0 is the only gate. Fold 1 runs once only if both Fold-0 correct count and
target-weighted accuracy improve.
