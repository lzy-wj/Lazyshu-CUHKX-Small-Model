# V82 V24 Conservative Rescue

V82 treats the public-best V24 prediction as an anchor and searches only for
high-consensus corrections. Thresholds and domain scope are selected on the
test-domain-weighted Fold 0, then transferred unchanged to Fold 1 and test.

The candidate expert pool contains V13, V23-base, V26, V29, V52, V57, V59,
and V71. Public labels and probe outcomes are not used by the search.
