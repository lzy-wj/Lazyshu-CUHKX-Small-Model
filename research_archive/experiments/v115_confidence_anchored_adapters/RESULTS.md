# Results

- V104 reference: 1175/1357 pair-aware, 89.25% target-weighted mean.
- V115 selected: 1173/1357 pair-aware, 88.98% target-weighted mean.
- Selected adapter weight: 0.2 from the fixed Fold0 grid.
- Decision: reject. Upweighting clean source labels reduces target adaptation;
  the V113 equal per-row objective transfers better despite noisier labels.
