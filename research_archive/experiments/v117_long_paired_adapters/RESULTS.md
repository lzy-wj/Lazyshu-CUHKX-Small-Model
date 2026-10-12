# Results

- V114: 1180/1357 pair-aware, 89.80% target-weighted mean.
- V117: 1177/1357 pair-aware, 89.47% target-weighted mean.
- Fold0 selects only 0.2 adapter weight after four epochs.
- Two/four-epoch probability soups also regress both folds.
- Decision: reject. Additional optimization improves clip top-1 but overfits
  target pseudo labels and damages source-template sequence likelihoods.
