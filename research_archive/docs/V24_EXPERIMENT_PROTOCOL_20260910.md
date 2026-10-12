# V24 Experiment Protocol

## Objective

Build one reproducible CNN/RNN/Transformer submission below `100,000,000` model-asset bytes that preserves V13 public robustness while transferring the verified V23 local gains. Public leaderboard feedback is used only for model-level validation, never to infer individual test labels.

## Frozen references

| Reference | Fold 0 | Fold 1 | Combined | Role |
|---|---:|---:|---:|---|
| V18 no-radar raw | 475/652 | 564/705 | 1039/1357 | Earlier single-model baseline |
| V23 warm-60 soup raw | 487/652 | 567/705 | 1054/1357 | Current single-model local best |
| V14 acceptance system | - | - | 1186/1357 | Previous system baseline |
| V23 second-stage system | 566/652 | 624/705 | 1190/1357 | Current local system best |

Public references are V13 `0.89552 = 180/201`, V14 pseudo `0.89054 = 179/201`, V14 fixed `0.88557 = 178/201`, and the rejected cross-fold V23 proxy `0.85074 = 171/201`.

## Split discipline

- Fold 0 holds out users `3, 8, 18, 23` and is the only architecture-selection fold.
- Fold 1 holds out users `5, 7, 17, 22` and is evaluated once, blind, for the Fold-0-selected candidate.
- For iterative refinement, train and compare `K=1/2/3` only on Fold 0. Select by correct predictions, then NLL, then visually incomplete-row accuracy. Train/evaluate only the selected `K` on Fold 1.
- Full-data training uses all `3036` labeled clips only after the architecture, iteration count, loss weights, epoch schedule, decoder and quantization recipe are frozen.

## Controlled variables

### Full Refitting

The architecture remains the V23 shared no-radar joint visual CNN plus temporal CNN/LSTM sensor encoders and Transformer fusion. The experiment changes only the fit set from a held-out fold to all labeled training data. Training-only teachers and parameter-soup sources do not count toward deployment assets, but the final prediction must be reproducible from the packaged assets.

### Iterative Refinement

Visual and sensor encoders execute exactly once. Their tokens are cached, and one shared refinement block is reused for `K` steps. `K=1` must reproduce the non-iterative control under the same data, seed, optimizer and losses. Step logits use deep supervision; soft feedback is detached unless a Fold-0-only ablation explicitly proves end-to-end feedback superior.

## Promotion gates

- Iterative refinement must beat its same-run `K=1` control on Fold 0 by at least three correct clips without worse NLL, then retain a positive combined gain after the single blind Fold 1 evaluation.
- A candidate replaces the V23 single-model reference only above `1054/1357` raw; it replaces the full local system only above `1190/1357` under a frozen decoder/router.
- Report complete-visual, missing-visual, paired, unpaired, per-user and worst-user accuracy. A gain caused solely by one user or one availability group is not sufficient for promotion.
- Final model assets must remain below `100,000,000` bytes. The design target is at most `95,000,000` bytes to retain packaging slack.
- Iterative inference is limited to `K <= 3`; encoders are not recomputed. Report measured latency and peak memory before deployment.

## Test-set policy

- Allowed: inference, synchronized test-time views, normalization-statistics adaptation, model-generated high-confidence consistency targets, and the frozen pair-aware sequence decoder.
- Disallowed: manual labeling, using ground-truth test answers, or submission patterns intended to recover individual public labels.
- Any transductive adaptation is an independent ablation and cannot replace the supervised full-data checkpoint unless it improves a predeclared local simulation and remains reproducible.

## GPU policy

- Inspect `nvidia-smi` immediately before every launch and never interrupt another user's process.
- From `08:00` to `18:00` Asia/Shanghai, leave at least four GPUs free.
- At night, use any genuinely idle GPUs. Each run writes its PID, GPU, command, start time and log path into its own experiment directory.

## Submission policy

V13 remains one final-submission slot. V24 receives the second slot only after all local gates, size checks, CSV checks and reproducibility checks pass. The public score is a secondary sanity check, not the architecture-selection metric.
