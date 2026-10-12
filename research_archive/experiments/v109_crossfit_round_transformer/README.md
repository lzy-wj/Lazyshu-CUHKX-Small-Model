# V109 Cross-Fitted Round Transformer

V109 learns sequence context without target-fold labels or pseudo-label
imitation. For Fold 0 evaluation it trains on Fold 1 OOF probabilities and
labels; for Fold 1 evaluation it trains on Fold 0 OOF probabilities and labels.
Those source users belong to the target fold's legal training set, while their
probabilities were produced by a model that excluded them.

The permitted two-layer Transformer consumes V89 and V104 normalized log
probabilities for every clip in an acquisition round. It predicts a bounded
residual over V104 and uses domain, position, and round-length embeddings.
Gaussian logit perturbation, dropout, anchor KL retention, and a soft uniqueness
penalty regularize the small cross-fitted sequence set. Training has a fixed
600-step endpoint and never selects settings on the target fold.
