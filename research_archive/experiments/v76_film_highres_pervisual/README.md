# V76 High-Resolution Per-Visual FiLM

V76 keeps the V65 joint-FiLM and per-visual auxiliary objective, but uses the
160-pixel person crop instead of the 128-pixel crop. Fold 0 and Fold 1 are
trained independently on physical GPUs 4 and 5. The candidate is promoted
only if the fixed sequence decoder improves both folds against V65.
