# Results

- V114 teacher: 1180/1357 pair-aware, 89.80% target-weighted mean.
- V116 iterative adapter: 1174/1357, 89.01% target-weighted mean.
- Fold0-selected fusion weight: 0.4.
- Decision: reject. The ten-row pseudo-label improvement is too sparse to
  survive a fresh parameter update and partly reinforces residual mistakes.
