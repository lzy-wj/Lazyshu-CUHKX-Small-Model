# V46 Fold-0 Results

The reproduced full-window prediction matches the historical V24 hard-adapted
Fold-0 raw score exactly at `493/652`.

The best temporal blends reach `552/652` after the frozen V32 decoder, only one
row above the `551/652` reference. Individual cropped views are all weaker than
the full clip, and the improvement is below the two-row Fold-0 gate. Fold 1 is
not run and synchronized temporal TTA is rejected.

