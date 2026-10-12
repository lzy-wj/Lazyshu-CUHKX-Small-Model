# V157 Partial-Script Stress Validation

Held-out local users contain roughly 160--190 clips, while each test user
contains only about 76--130. The standard folds therefore validate complete
triplicate scripts, but deployment sees visibly partial and unequal scripts.

V157 creates five deterministic 70% clip-retention views of each held-out fold
without changing labels or training. It recomputes V127 and the V154
triplicate-capacity decoder on every partial view. This is a stress audit, not
a new selection fold: promotion requires a consistent gain across seeds and
both folds, not a favorable average alone.
