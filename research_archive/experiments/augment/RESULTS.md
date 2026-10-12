# Fold-0 Result: Async Temporal Shift

## Decision

Neither screened magnitude passes the preregistered fold-0 gate. Do not advance this augmentation to fold 1 and retain the frozen V9 baseline.

## Controlled Variable

The video stream is the temporal anchor. During training only, skeleton and IMU encoder tokens are shifted independently for every sample by a uniformly sampled integer offset in `[-k, k]`. Edge tokens are replicated, validation is unshifted, and modality availability masks are unchanged. All architecture, initialization, preprocessing, optimizer, seed, budget, soup, TTA, blend and decoder settings are frozen in `baseline_audit.json`.

## Metrics

| Candidate | 32f right endpoint | Soup standalone | Available-only | Fixed V9 raw | Fixed decoder | Gate |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 65.3374% | 67.7914% | 69.4969% | 71.0123% | 76.8405% (501/652) | Reference |
| `±2` tokens | 65.4908% | 68.4049% | 70.1258% | 70.8589% | 76.3804% (498/652) | Fail |
| `±4` tokens | 65.0307% | 67.0245% | 68.7107% | 69.6319% | 76.9939% (502/652) | Fail |

The `±2` candidate improves standalone accuracy by `+0.6135pp` and available-only accuracy by `+0.6289pp`, but loses `0.1534pp` in the fixed V9 blend and three decoded samples. The `±4` candidate loses `0.7669pp` standalone and `1.3804pp` in the blend; its one additional decoded sample is below the two-sample gate and comes with a raw regression.

## Interpretation

Moderate asynchronous shifting acts as useful standalone regularization, but it reduces the error complementarity that made the R(2+1)D branch valuable inside V9. A four-token window is too aggressive for this encoder. Plain shift augmentation alone is therefore not selected; explicit cross-temporal attention remains the more plausible mechanism for handling real misalignment.

## Reproduction

```bash
bash experiments/augment/scripts/run_fold0.sh 2
bash experiments/augment/scripts/evaluate_fold0.sh 2
bash experiments/augment/scripts/run_fold0.sh 4
bash experiments/augment/scripts/evaluate_fold0.sh 4
```

SwanLab runs:

- `±2`: `https://swanlab.cn/@shu_123/cuhkx-small-multimodal/runs/bj3ugsmx`
- `±4`: `https://swanlab.cn/@shu_123/cuhkx-small-multimodal/runs/8dnjx0dq`

Candidate prediction files:

- `outputs/predictions/ablation_async_shift_f0_p2_candidate.npz`, SHA256 `0da8c9ff515a0ceab7b4302f7f2640d711877f5f9b72ab906d70ed65ada40d78`
- `outputs/predictions/ablation_async_shift_f0_p4_candidate.npz`, SHA256 `a15220c6f2490545ea633b9a5aed25b4151779517500dd69c65d829c652f8beb`

Both files contain exactly 652 fold-0 rows with `probabilities`, `labels` and `indices`; their indices and labels were asserted equal to the unified scorer reference. Fold 1 was not inspected, and no Kaggle submission was made.
