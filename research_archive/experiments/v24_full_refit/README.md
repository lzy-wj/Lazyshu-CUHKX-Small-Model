# V24 Full-Data Refit

This directory implements the full-data branch of `docs/V24_EXPERIMENT_PROTOCOL_20260910.md`. It does not perform architecture selection, consume test labels, or submit to Kaggle.

## Frozen protocol

- Train on all `3036` labeled clips with `--fold -1` only after the V23 recipe is frozen.
- Preserve the V23 shared no-radar model: joint Depth/Thermal/IR R(2+1)D CNN, temporal CNN/LSTM sensor encoders, and Transformer fusion.
- Use effective random seed `42`. The shared trainer seeds with `args.seed + args.fold`, so the full-data runners pass `--seed 43` to compensate for `fold=-1`.
- Fix source epochs from Fold-0 selection: no-radar `13`, cross-user `16`, progressive-160 `7`, scratch mask-distill `16`, and warm mask-distill `9`.
- Fix soup source weights to `0.082416/0.079504/0.148508/0.089572/0.600000`, selected on Fold 0 by accuracy then NLL.
- Use the existing pair-aware decoder unchanged. It uses training-label transitions/templates and test acquisition structure, never test labels.

## Source decision

All five V23 sources are retained. Progressive-160 and cross-user underperform as standalone models but supplied nonzero Fold-0-selected diversity weights. Scratch mask-distill is weaker than warm mask-distill, yet removing its `0.089572` weight would create an unvalidated four-source recipe. Full-data fitting is therefore a strict refit of the proven V23 soup rather than a new model-selection round.

The 160-pixel source is trained at 160 then parameter-averaged into a soup whose inference configuration comes from the 128-pixel no-radar source. This exactly follows the validated V23 construction.

## Training schedule

The base, cross-user, and scratch mask-distillation runs are independent. Progressive-160 and warm mask-distillation initialize from the completed full-data base. Full-data has no validation loader, so each fixed epoch endpoint becomes `best.pt`; these runs must not be used to tune epoch count or soup weights.

Night-only orchestration, after checking that GPUs are idle:

```bash
tmux new -s v24-full
cd ${LAZYSHU_PROJECT_ROOT}
SWANLAB_MODE=cloud experiments/v24_full_refit/run_night_pipeline.sh 0 1 3 4 5
```

Individual runs are also reproducible:

```bash
experiments/v24_full_refit/run_full_no_radar.sh GPU
experiments/v24_full_refit/run_full_cross_user.sh GPU
experiments/v24_full_refit/run_full_scratch_maskdistill.sh GPU
experiments/v24_full_refit/run_full_progressive160.sh GPU
experiments/v24_full_refit/run_full_warm_maskdistill.sh GPU
```

Every GPU runner refuses a busy GPU, enforces four idle GPUs during `08:00-18:00` Asia/Shanghai, and records PID, GPU, command, timestamps, and log path under `run_metadata/`.

## Finalization

After all five training completions:

```bash
experiments/v24_full_refit/run_finalize.sh GPU
```

Finalization builds FP32, INT8, and hybrid INT4 soups; predicts the 405 test clips with the primary INT8 model; applies the frozen pair-aware decoder; verifies CSV ordering and class range; verifies every source is a `3036`-clip full fit; verifies source hashes/weights; and enforces a `95,000,000`-byte deployment target. It creates a local submission CSV but never invokes Kaggle.

Primary deployment assets are one INT8 soup plus compact YOLO, estimated at `70,942,739` bytes. The hybrid INT4 fallback is estimated at `41,959,066` bytes. Actual sizes replace estimates in `outputs/reports/release_audit.json`.

## Validation

Static checks do not launch GPU work:

```bash
experiments/v24_full_refit/validate_static.sh
```

Expected outputs are enumerated in `expected_artifacts.json`; the frozen machine-readable recipe is `protocol.json`; asset accounting is in `asset_budget.json`.

## Risks

- Full-data endpoints have no honest validation score; success must not be inferred from training accuracy.
- The public V23 failure shows that cross-fold parameter averaging is not a substitute for this refit, but does not guarantee the full refit will recover V13 public performance.
- Kinetics initialization remains an external model asset during training. It is not needed for inference, but competition provenance rules must still allow the already-used initialization.
- The shared trainer's `seed + fold` behavior is compensated in runners rather than changed globally because this agent may not edit shared code.
- INT4 retained ten fewer raw validation clips than INT8 in V23 and is only a size fallback, not the primary candidate.
