# Async Temporal Shift Ablation

This fork changes only training-time temporal alignment. Video tokens remain the temporal anchor; skeleton and IMU encoder tokens receive independent per-sample integer offsets sampled uniformly from `[-k, k]`. Boundary values are replicated rather than wrapped, modality masks are untouched, and shifts are disabled in evaluation mode.

The model architecture, fold, visual initializer, cache, optimizer, random seed, training budget, weight-soup coefficient, four-view TTA, V9 blend and repeat decoder are frozen to `baseline_audit.json`.

Run fold-0 screening with:

```bash
bash experiments/augment/scripts/run_fold0.sh 2
bash experiments/augment/scripts/evaluate_fold0.sh 2
```

All GPU work is serialized through `outputs/logs/gpu-2.lock`. Checkpoints, predictions, reports, logs and SwanLab files are written below `experiments/augment/outputs`.

Fold-0 screening is complete. Neither `±2` nor `±4` passes the acceptance gate; see `RESULTS.md` and `results_summary.json`.
