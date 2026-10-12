# V11 Cross-Subject Generalization Study

## Outcome

- The verified production system remains V9 with Kaggle public score `0.82089` and subject-disjoint decoded accuracy `80.6927%` across the two historical folds.
- V11 trained a same-code control and eight completed generalization candidates. The inference architecture remained the same 64,955,492-parameter CNN/LSTM/Transformer model.
- Several methods improved representation-level or fold-0 metrics, but none improved the frozen final system on fold 1. No V11 full-data refit, package replacement, or Kaggle submission was made.
- `outputs/final_legal_multimodal_v9.zip` remains unchanged at 84,611,833 bytes, SHA256 `6093c916b24fb851ac8aa3dd8296872f9a72aeb20c2e4066d017f51e5de1c6c7`.

## Scientific Controls

- Fold 0 validation users were `[3, 8, 18, 23]`; fold 1 users `[5, 7, 17, 22]` were not used until the fold-0 selection was frozen.
- The same-code control reproduced the training setup but reached only `65.4908%` standalone and `68.7117%` final raw, versus historical V9 `67.7914%` and `71.0123%`. The exact historical 32-frame source snapshot is unavailable, so all conclusions use both the control and historical V9 as references.
- An independent implementation review stopped the first four-epoch pilots. It found that the original visual dropout was a no-op in non-shared mode, an unbalanced SupCon queue was too strong, and an unconditional user adversary could erase action information because user and class support are confounded.
- Confirmatory code therefore preserved the zero-method RNG path, added balanced queue limits and warmup, made dropout symmetric over video/skeleton/IMU, and conditioned the adversary on action labels.
- All checkpoint selection remained clean single-view EMA accuracy. Final soup weight, four-view TTA, base ensemble, X3D-L gate, and repeat decoder were frozen except for the explicitly documented fold-0 rescue search.

## Fold-0 Results

| Candidate | Best 32f endpoint | Standalone soup | V9 raw | Decoded | Decision |
|---|---:|---:|---:|---:|---|
| Historical V9 | 65.3374% | 67.7914% | 71.0123% | 76.8405% (501) | Keep |
| Same-code control | 63.9571% | 65.4908% | 68.7117% | 74.6933% (487) | Reference only |
| Symmetric modality dropout | 64.1104% | 65.3374% | 68.8650% | 75.0000% (489) | Reject |
| Balanced cross-user SupCon | 64.4172% | 67.9448% | 69.6319% | 76.3804% (498) | Reject |
| Conditional user adversary | 65.4908% | 68.2515% | 71.1656% | 74.8466% (488) | Fold-0 raw signal |
| Frozen backbone BN | 65.6442% | 67.9448% | 70.5521% | 76.2270% (497) | Reject |
| Backbone BN momentum `0.1` | 65.3374% | 67.0245% | 68.8650% | 75.3067% (491) | Reject |
| EMA decay `0.999` | 65.4908% | 66.5644% | 68.7117% | 76.2270% (497) | Reject |
| Synchronized speed jitter `[0.8, 1.2]` | 66.1043% | 67.3313% | 70.2454% | 75.9202% (495) | Reject |

Inverse-square-root class balancing selected a `63.6503%` endpoint, below the same-code control, and was stopped without a final four-view evaluation. Synchronized `[0.8, 1.2]` speed jitter later completed after a GPU became free; it improved NLL but regressed standalone, raw, decoded, macro-class and worst-user accuracy, so Fold 1 was not run.

## Generalization Diagnostics

### Cross-User SupCon

- Standalone accuracy improved by `+2.4540` points over the same-code control and `+0.1534` points over historical V9.
- Worst-user standalone accuracy increased from historical `65.0307%` to `65.6442%`, while between-user standard deviation fell from `2.6437%` to `1.4899%`.
- NLL improved from `1.7985` to `1.6694`, but fixed V9 ensemble raw accuracy fell by `1.3804` points. SupCon made the branch more uniform but less complementary to the existing base models.

### Conditional User Adversary

- Fold 0 standalone improved over historical V9 by `+0.4601` points, user-macro by `+0.4643` points, worst-user by `+0.2176` points, and NLL by `0.1593`.
- The direct fixed V9 blend improved raw by one sample but lost thirteen decoded samples, showing strong calibration/sequence incompatibility.
- A 50/50 weight soup of historical and adversarial R2 models reached `68.7117%` standalone. A fold-0-only search froze R2 weight `0.425` and temperature `1.125`, producing `77.4540%` decoded (`505/652`, +4 samples).

### BatchNorm and EMA

- Freezing BN and lowering its momentum improved selected single-view endpoints and often NLL, confirming that batch-8 running statistics are noisy.
- Their four-view soups did not preserve the gains. Long EMA substantially reduced user variance but also reduced macro-class and aggregate accuracy.
- These results show that a stronger validation endpoint alone is not sufficient; the 16/32-frame soup and TTA response must remain complementary.

### Speed Jitter

- The completed speed-jitter soup reached `67.3313%` standalone, `70.2454%` final raw and `75.9202%` decoded (`495/652`).
- NLL improved by `0.1029`, but macro-class accuracy fell `2.2131` points and worst-user accuracy fell `2.6193` points. The augmentation changed confidence without improving cross-user decisions.

## Fold-1 Confirmation

The fold-0 rescue configuration was frozen in `experiments/generalization_v11/selected_config.json` before fold 1.

| Metric | Fold 0 baseline | Fold 0 selected | Fold 1 baseline | Fold 1 selected |
|---|---:|---:|---:|---:|
| Standalone | 67.7914% | 68.7117% | 71.9149% | 70.6383% |
| Final raw | 71.0123% | 70.5521% | 79.1489% | 78.7234% |
| Decoded | 76.8405% (501) | 77.4540% (505) | 84.2553% (594) | 81.5603% (575) |

- Fold 1 rejected the selection: standalone fell `1.2766` points, final raw fell `0.4255` points, and decoded lost 19 samples.
- Weighted decoded accuracy was `79.5873%` (`1080/1357`) versus V9 `80.6927%` (`1095/1357`).
- The fold-0 gain was therefore selection-specific rather than a deployable cross-subject improvement.

## Interpretation

1. The main problem is not simply insufficient regularization. Several interventions improved NLL, worst-user accuracy, or single-view endpoints without improving the final ensemble.
2. V9 depends heavily on error complementarity among the 16-frame branch, 32-frame branch, lightweight base, X3D-L gate, TTA, and repeat decoder. A branch can become individually better while making the system worse.
3. User 5 remains the dominant fold-1 standalone failure, but the V9 base ensemble already repairs much of it. Optimizing only the R2 representation cannot reproduce that complementarity reliably.
4. The next useful experiment should optimize robustness and complementarity together: synchronized speed jitter followed by out-of-fold distillation from the full V9 probability ensemble into one R2 student. Plainly increasing loss regularization is unlikely to produce a large jump.

## Reproducibility

- Protocol and commands: `experiments/generalization_v11/README.md`.
- Independent review: `experiments/generalization_v11/review/REVIEW.md`.
- Method audits: `experiments/generalization_v11/audit_methods.py`.
- Canonical fold reports: `experiments/generalization_v11/outputs/reports`.
- Fold-0 frozen selection: `experiments/generalization_v11/selected_config.json`.
- Fold-1 reproduction: `experiments/generalization_v11/build_selected_fold.sh`.

SwanLab runs: control `romdz3tm`; structured dropout `m967rgeh`; SupCon `m50j91s4`; conditional adversary fold 0 `y5ptdmm4`, fold 1 `63adl6zm`; frozen BN `g8cbbz7q`; BN momentum `9w7g2hmz`; class balance `h8040ul4`; EMA `d0p5y7il`; speed jitter `dk8zoeh2`.
