# V139 Environment-Specific Paired Strength

V114 uses one paired-person adapter weight for both acquisition environments.
V139 allows the low-ID and high-ID paired domains to use separate weights,
while preserving the same adapters, probabilities, and sequence decoder.
The two weights are selected jointly on Fold 0 from a coarse grid and then
transferred unchanged to Fold 1.
