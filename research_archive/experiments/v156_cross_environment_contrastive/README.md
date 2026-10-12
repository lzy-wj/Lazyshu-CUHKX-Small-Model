# V156 Cross-Environment Contrastive Alignment

Hard environment isolation in V148 improves standalone evidence but conflicts
with V127. V156 keeps all training data and instead applies a soft supervised
domain-generalization objective: same-action features from opposite physical
environments are positives, while different actions are negatives. Same-action
examples from the same environment are excluded from the contrastive denominator.

The run warm-starts V23 and changes only this representation regularizer. A
small 0.01 weight and the established bounded class queue are fixed in advance.
Fold 1 is allowed only after a Fold-0 system-level gain.
