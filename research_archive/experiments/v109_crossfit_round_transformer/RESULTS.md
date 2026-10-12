# Results

| Candidate | Raw | Pair-aware | Target-weighted pair |
| --- | ---: | ---: | ---: |
| V104 | 1,076/1,357 | **1,175/1,357** | **89.2466%** |
| V109 cross-fitted Transformer | 1,052/1,357 | 1,144/1,357 | 87.5437% |

The model fits its source OOF rounds strongly but transfers the wrong script
and confusion corrections to unseen users. This is not a capacity failure; it
is a cross-user conditional-shift failure. The existing explicit paired-user
decoder is retained, and learned round correction is closed.

The full audit is in `outputs/evaluation.json`.
