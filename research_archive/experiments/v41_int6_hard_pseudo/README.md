# V41 INT6 Hard Pseudo

V41 uses group-wise 4-bit storage for the deepest R(2+1)D layer, 6-bit storage
for layer 3, and INT8 for early/fusion/sensor layers. The expected checkpoint
size is about 41 MB, leaving enough room for the 43.9 MB V14 compact backbone.

Fold 0 must retain at least `550/652` direct pair-aware accuracy before Fold 1
or any dual-backbone deployment is allowed.
