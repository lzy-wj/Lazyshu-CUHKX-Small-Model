# V151 Hierarchical Activity Loss

The official CUHK-X ontology partitions the 40 labels into seven contiguous
activity families: Personal Care, Eating & Drinking, Household, Working,
Socializing & Leisure, Sports & Exercises, and Caring & Helping.

V127 still makes 80 cross-family errors among its 152 OOF mistakes. V151 adds
an exact coarse-category negative log-likelihood obtained by summing the fine
class probabilities inside each official family. It adds no parameters and
keeps the complete V23 architecture, data, and augmentation recipe fixed.

One conservative coarse loss weight of 0.2 is preregistered on Fold 0. Fold 1
is run only if the final V127-aware Fold-0 gate improves.
