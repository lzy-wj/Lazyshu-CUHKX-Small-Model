# V12 Blend-Aware Multimodal Study

## Outcome

- Conservative deployable local result: `1103/1357 = 81.2822%`, versus V9 `1095/1357 = 80.6927%` (`+8` correct, `+0.5895` pp).
- Best local result with the fold-0-selected unequal-round alignment decoder: `1107/1357 = 81.5770%` (`+12` correct, `+0.8843` pp over V9).
- The selected R2 system stores one INT8 Mixup model plus a `1,066,294`-byte rank-4 adapter that reconstructs a regularized second expert in memory. Projected model assets are `98,386,739` bytes, below 100 MB by `1,613,261` bytes.
- Fold 0 selected rank `4` and old-expert weight `0.125`; Fold 1 was evaluated once with these values frozen. Fold 1 reached `561/705` raw and `590/705` decoded.
- No Kaggle submission was made.

## Selected Model

1. R(2+1)D-34 encodes 32-frame Depth_Color plus IR video.
2. Temporal CNN encodes skeleton; CNN plus bidirectional LSTM encodes world-frame IMU.
3. A two-layer Transformer fuses the three token sequences with missing-modality masks.
4. Synchronized multimodal Mixup with `alpha=0.3` trains the main model.
5. The original V9-like expert is represented as a rank-4 quantized low-rank delta from the Mixup model, not as a second 65 MB checkpoint.
6. Candidate probabilities use `0.875 × Mixup + 0.125 × reconstructed expert`, followed by the frozen `0.4 × base + 0.6 × R2(T=2)` blend and X3D-L low-confidence gate.
7. Repeated rounds use power `1.5`, environment-specific transition scores and beam-100 unique-label decoding.

## Two-Fold Result

| System | Fold 0 | Fold 1 blind | Combined |
|---|---:|---:|---:|
| V9 | 501/652 | 594/705 | 1095/1357 (80.6927%) |
| Single Mixup, repeat power 1.5 | 513/652 | 587/705 | 1100/1357 (81.0612%) |
| Rank-4 dual expert, standard decoder | 513/652 | 590/705 | **1103/1357 (81.2822%)** |
| Rank-4 dual expert, aligned decoder | 513/652 | 594/705 | **1107/1357 (81.5770%)** |

The aligned decoder gained four Fold-1 samples from two genuinely repeated groups with one missing action. The current 405-row test set has no unequal-length repeat group under the frozen detector, so this is a valid local robustness gain but is not expected to change the current test prediction by itself.

## Main Findings

### Mixup

- Seed-42 Mixup improved final raw accuracy by 13 samples across two folds and was the first single-R2 candidate above V9 after decoding.
- A second seed did not replicate the gain. This justified retaining one main model and compressing a complementary direction rather than packaging a two-seed ensemble.

### Fusion Diagnosis

- On Fold 0, the selected fused model and its video auxiliary branch differed on only two of 652 predictions.
- Skeleton alone correctly classified 14 fused-model errors and IMU correctly classified 12, proving useful information existed but the residual fusion head rarely overruled video.
- Class-wise expert gating improved the matched 15% modality-dropout control, but not the original Mixup baseline. The limitation was training distribution and complementarity, not Transformer depth alone.

### Six Modalities

- Shared six-modal Mixup reached the strongest single-view endpoint, `70.0920%` at epoch 18, and `70.3988%` under four-view TTA.
- Its frozen final blend reached only `467/652` raw and `501/652` decoded. A three-expert simplex search selected its weight as zero.
- Radar is usable in only `1409/3036` labeled clips and all six modalities coexist in `1287/3036`; treating radar as a universal stream is therefore inappropriate.

### Rejected Changes

| Change | Best relevant Fold-0 result | Decision |
|---|---:|---|
| Six-modal shared encoder + Mixup | 70.3988% standalone; 501 decoded | Reject from final blend |
| Strong auxiliary supervision | 69.9387% standalone; 473 raw; 506 decoded | Reject |
| Seven-channel early visual fusion | 67.6380% standalone; 464 raw; 493 decoded | Reject |
| Six-layer fusion Transformer | 468 raw; 494 decoded | Reject |
| Canonical skeleton pretraining | 66.5644% endpoint | Reject |
| MixStyle | 64.1104% endpoint | Reject |
| Temporal embedding | 504 decoded | Reject |
| 15% modality dropout | 66.4110% endpoint | Reject |
| Gated 15% modality dropout | 66.7178% endpoint | Reject |
| Position-conditioned sequence prior | selected strength `0` | Reject |

## Compression

- The two selected fold models have state cosine similarity `0.9869`; their difference norm is only `16.3%` of the old state norm.
- Rank-4 randomized SVD is applied per matrix/conv tensor after flattening output channels. Left and right factors use row-wise symmetric INT8; one-dimensional deltas remain FP16.
- Rank-4 adapter size is `1,066,294` bytes. Rank 8/12/16 sizes are `1,692,726`, `2,319,094` and `2,945,590` bytes.
- Rank 4 gave the best final Fold-0 result: `476` raw and `513` decoded. Higher rank reduced state reconstruction error but did not improve decisions.
- Group-wise INT4 was also implemented and audited. Even group size 64 lost too much probability geometry; low-rank delta compression was both smaller and more accurate.

## Evidence

- Final two-fold result: `experiments/blend_aware_v12/outputs/lowrank_delta_blind_result.json`.
- Aligned-decoder result: `experiments/blend_aware_v12/outputs/lowrank_aligned_decoder_blind.json`.
- Fold-0 rank selection: `experiments/blend_aware_v12/outputs/lowrank_delta_f0_search.json`.
- Mixup baseline: `experiments/blend_aware_v12/outputs/v12_mixup_seed42_summary.json`.
- Six-modal evaluation: `experiments/blend_aware_v12/outputs/evaluations/v12_shared6_mixup_e18_f0.json`.
- Modality availability: `outputs/study_scratch/availability_audit.json`.
- Adapter builder: `experiments/blend_aware_v12/build_low_rank_delta.py`.
- Adapter-aware inference: `scripts/predict_r2plus1d.py`.

## Remaining Delivery Step

The full-data Mixup model is trained for 10 epochs, matching the mean selected update budget of Fold 0 epoch 9 and Fold 1 epoch 16. It is then mixed `0.25/0.75` with the historical 16-frame full model, compacted to INT8, and paired with a rank-4 delta to the V9 full R2 soup. Final test inference and the exact serialized model-asset audit must complete before this V12 system is considered packaged.
