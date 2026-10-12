# V15 Generalization Result and V16 Architecture Plan

## Frozen reference

- Current public best: V13 pair-aware, `0.89552` (`180/201`).
- Current local best: V14 pseudo-adapted soup with the fold-0-selected acceptance gate, `1186/1357` (`87.40%`). The ungated pseudo result is `1185/1357`.
- V14 fixed soup reached `1179/1357` locally but only `0.88557` publicly.
- No further public submission is justified without a material two-fold improvement.

## V15 cross-user regularization

| Candidate | Fold 0 | Fold 1 | Decision |
|---|---:|---:|---|
| Cross-user contrastive, weight 0.05 | 69.79% | 76.74% | Keep only as a diversity source |
| Cross-user contrastive, weight 0.02 | 68.10% | not run | Reject |
| MixUp + cross-user contrastive | 69.79% | not run | Reject as direct replacement |
| MixUp baseline | 70.09% | 73.48% | Reference |
| Conditional subject adversary, weight 0.02 | 68.87% | not run | Reject |

Cross-user contrastive improves fold 1 by 3.26 points over the MixUp endpoint but is 0.31 points lower on fold 0. Its light fold-0 blend corrected nine additional raw samples, so it remains useful for later diversity evaluation, not as a proven replacement.

## V16 root-cause hypothesis

The shared visual encoder processes Depth, Thermal and IR sequentially through one R(2+1)D network. Every BatchNorm layer uses momentum `0.9`; therefore each training step updates statistics in a fixed Depth→Thermal→IR order. The final update is approximately 90% IR, leaving the evaluation statistics systematically biased toward the last modality.

This is an architectural normalization defect rather than a scalar hyperparameter issue. V16 keeps the V14 MixUp protocol fixed and compares:

1. `joint`: concatenate the three modalities on the batch axis so every shared BatchNorm update sees an equal mixture.
2. `modality_bn`: share convolution weights but maintain independent BatchNorm state for each modality.
3. `separate_stem_joint`: use modality-specific low-level stems, then a shared trunk whose BatchNorm sees all modalities jointly.

All three candidates run on both user-disjoint folds. Selection requires a combined improvement and no severe fold regression.

## Next independent direction

The fusion Transformer currently receives four temporal tokens from every visual stream and sixteen tokens from each structured stream. With no temporal embedding it cannot directly identify corresponding action phases. The rejected index-embedding pilot was also misaligned: visual indices 0–3 were paired with only the first four sensor positions instead of spanning the full 0–15 phase range.

The next candidate interpolates one shared sixteen-step embedding over each stream's complete normalized duration. Thus four visual tokens represent phases `[0, 1/3, 2/3, 1]`, while sixteen sensor tokens cover the same endpoints. This remains isolated until V16 chooses the visual normalization strategy.

## Frozen error audit

The reproducible report is `experiments/scene_context_v13/outputs/v13_error_structure.json`.

- V13 reaches `633/699 = 90.56%` on paired rows but only `528/658 = 80.24%` on unpaired rows.
- Complete visual input reaches `1074/1227 = 87.53%`; clips missing at least one visual stream reach only `87/130 = 66.92%`.
- All six modalities reach `491/553 = 88.79%`; missing-any rows reach `670/804 = 83.33%`.
- The largest unpaired confusion remains Read documents versus Turn pages, requiring local visual-temporal detail rather than more sequence-template tuning.

## Literature check

- Domain-Specific Batch Normalization shares convolutional parameters while separating normalization state across domains, matching the V16 modality-BN hypothesis: <https://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html>.
- Temporal Cycle-Consistency Learning supports learning dense phase correspondence across sequences: <https://openaccess.thecvf.com/content_CVPR_2019/html/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.html>.
- COCOA reports label-efficient cross-modality contrastive learning for multisensor data: <https://arxiv.org/abs/2208.00467>.
- MultiMAE supports masked cross-modal predictive pretraining and remains a later option if supervised architecture changes saturate: <https://arxiv.org/abs/2204.01678>.

Priority remains controlled: V16 normalization first, normalized phase attention second, and cross-modal masked/contrastive pretraining only if both supervised changes fail.

An additional supervised representation check replaces the averaged visual auxiliary loss with the mean loss over available Depth, Thermal and IR streams. The total auxiliary coefficient is unchanged; this tests whether direct per-stream semantic supervision prevents one visual modality from hiding another's errors.

## V16 completed results

| Visual strategy | Fold 0 | Fold 1 | Combined endpoint |
|---|---:|---:|---:|
| Previous sequential MixUp | 70.09% | 73.48% | 975/1357 (71.85%) |
| Joint modality batch | **72.09%** | **78.01%** | **1020/1357 (75.17%)** |
| Modality-specific BN | 71.78% | 77.45% | 1014/1357 (74.72%) |
| Separate stems, joint trunk | 71.78% | 76.60% | 1008/1357 (74.28%) |

Joint batching is the selected V16 backbone because it improves both folds and adds 45 correct endpoint predictions without adding parameters. The result supports the fixed-order BatchNorm diagnosis. With momentum `0.9`, a sequential Depth→Thermal→IR update has approximate old/depth/thermal/IR weights `0.001/0.009/0.09/0.9`; joint batching removes this order bias.

Under a newly selected V13 candidate gate, the joint model reaches `1177/1357 = 86.74%` after pair-aware decoding, versus V13 `1161/1357`. It does not yet exceed the V14 fixed soup (`1179`) or pseudo-adapted result (`1185`). A 50/50 probability ensemble of the old fixed-soup branch and V16 joint branch reaches `1033/1357 = 76.12%` before system blending, demonstrating useful error diversity, but the current V13 decoder integration reaches only `1173/1357`.

V17 therefore tests two representation changes on both folds: continuous phase embeddings with and without joint batching, plus per-visual-stream semantic supervision on the selected joint backbone.

## V16 deployment compression check

A 50/50 cross-strategy parameter soup averages the old sequential fixed-soup state with the V16 joint state, then performs one INT8 quantization pass and executes with joint visual batching. It is `65,610,802` bytes and reaches `475/652` plus `552/705`, or `1027/1357 = 75.68%`, seven endpoint samples above V16 joint while remaining a single model.

The current V13 pair-aware integration reaches only `1166/1357`, so it is not promoted over V14. Replacing the main R2 branch also reaches `1166/1357`. The compression itself succeeds; the remaining issue is compatibility with the older base/X3D/sequence ensemble, not loss of the V16 representation gain.

## V18 planned direction

The CUHK-X paper reports that class-balanced resampling raises its RGB random-split baseline from `90.89%` to `96.16%`. Our previous inverse-square-root class-weighted loss is not equivalent to resampling and was rejected on the older backbone. V18 will use a uniform-over-class replacement sampler with V16 joint batching and unchanged MixUp, loss coefficients, epoch budget and architecture. This directly changes rare-class exposure without increasing deployment size.

## V17 completed results

| Candidate | Fold 0 | Fold 1 | Combined endpoint | Decision |
|---|---:|---:|---:|---|
| V16 joint reference | 72.09% | 78.01% | 1020/1357 | Keep |
| Joint + continuous phase | 71.78% | 78.01% | 1018/1357 | Reject |
| Joint + per-visual auxiliary | 71.47% | 78.16% | 1017/1357 | Reject |
| Sequential + continuous phase | 69.02% | 72.62% | 962/1357 | Reject |

The phase hypothesis is plausible architecturally but does not add subject-held-out accuracy. Per-stream supervision shifts one sample from fold 0 to fold 1 and also fails the combined criterion.

V18 adds two independent controls alongside balanced sampling. One removes training-time horizontal flips because only video is flipped while skeleton and IMU retain their original orientation. The other removes radar because it is usable in only `1409/3036` labeled clips. Both retain every other V16 setting.

## V18 completed results

| Candidate | Fold 0 | Fold 1 | Combined endpoint | Decision |
|---|---:|---:|---:|---|
| V16 joint reference | 72.09% | 78.01% | 1020/1357 | Reference |
| Class-balanced sampling | 71.17% | 77.87% | 1013/1357 | Reject |
| No horizontal flip | 69.63% | 75.46% | 986/1357 | Reject |
| Joint without radar | **72.85%** | **80.00%** | **1039/1357** | Keep |

Removing radar improves both held-out folds and adds 19 correct predictions over V16 joint and 64 over the old sequential six-modality MixUp endpoint. This is consistent with radar being both the sparsest modality and the weakest single-modality signal in the dataset. Flip/crop TTA leaves top-1 accuracy unchanged on both folds. The INT8 checkpoint is `65,383,115` bytes, so the selected branch remains under the deployment limit.

The gain does not transfer directly to the full V13 system. The frozen fold-0-selected, fold-1-blind domain pair-aware integration reaches `1175/1357 = 86.59%`, versus V13 `1161`, V14 fixed soup `1179`, and V14 pseudo adaptation `1185`. Against the existing V13 raw ensemble, the no-radar branch has 44 candidate-only versus 45 base-only correct samples on fold 0 and 42 versus 39 on fold 1. Their raw totals are almost equal (`1039` versus `1037`), while their oracle union is `1123`; the next bottleneck is therefore reliable sample-wise routing rather than another global blend weight.

## Fine-detail control

The existing person-crop X3D-S run at `224x224` reaches `438/652 = 67.18%` and `513/705 = 72.77%`. Under the same frozen domain pair-aware protocol it returns exactly `1161/1357`, so extra crop resolution alone does not improve the current system. This rejects resolution scaling as the next main direction, while retaining the branch as a diversity source.

## V19 and V20 completed directions

V19 tests two structural generalization hypotheses on the selected no-radar joint backbone: a soft environment-conditioned residual mixture-of-experts and cross-user supervised contrastive learning. Both are evaluated on both user-disjoint folds.

V20 addresses premature temporal compression. The current R(2+1)D fusion sees only four final-stage tokens from each visual modality. The visual-pyramid candidate concatenates eight layer-3 tokens with four layer-4 tokens per visual stream before cross-modal Transformer fusion. It adds only one `256 -> 192` projection (about 49 thousand parameters), for `65,003,015` total parameters, and changes no optimization setting relative to V18.

| Candidate | Fold 0 | Fold 1 | Combined endpoint | Pair-aware integration |
|---|---:|---:|---:|---:|
| V18 no-radar reference | 72.85% | 80.00% | 1039/1357 | 1175/1357 |
| Environment-conditioned MoE | 71.47% | 78.01% | 1016/1357 | 1164/1357 |
| Cross-user contrastive | 71.63% | 78.44% | 1020/1357 | 1170/1357 |
| Visual temporal pyramid | 70.09% | 79.29% | 1016/1357 | 1173/1357 |

Neither V19 objective improves the direct endpoint or the frozen pair-aware system. The visual pyramid improves fold 1 relative to V16 joint but regresses fold 0, and its integration also remains below V18.

A probability ensemble with weights `0.50/0.25/0.25` for no-radar, cross-user, and visual-pyramid models is selected as the minimum-NLL option among the fold-0 top-1 ties on a fixed 0.25 simplex. Fold 1 remains blind. It reaches `475/652` and `567/705` raw, then `1177/1357` under domain pair-aware decoding. This is the best integration among the new branches but remains nine samples below the gated V14 local best.

## V21 active directions

An activation audit of the V18 checkpoint over 200 fold-0 samples finds mean pre-fusion token norms of `39.18` for visual, `20.61` for skeleton, and `7.22` for IMU. V21 tests independent per-modality LayerNorm before modality embeddings and cross-modal attention. This adds only 1,920 parameters.

A new three-visual cache preserves 32 frames while increasing the complete scene crop from `128x128` to `160x160`. Unlike the earlier 224-pixel detail branch, it retains Depth, Thermal, IR, skeleton, and IMU. A second controlled candidate increases cross-modal Transformer depth from two to four layers. Both retain V18's no-radar joint backbone and training protocol.

The completed token-normalization candidate reaches `461/652 + 551/705 = 1012/1357`; its pair-aware integration is `1172/1357`. Four-layer fusion reaches `470/652 + 544/705 = 1014/1357` and integrates to `1167/1357`. Both are rejected. The large input-norm mismatch is real, but pre-normalized Transformer attention already limits its effect; additional input LayerNorm does not improve held-out users.

The 160-pixel scene candidate remains active. V22 additionally tests EMA teacher consistency under a randomly removed available modality, directly targeting missing-input robustness without adding deployment parameters. A seed-43 reproduction of V18 runs in parallel to test whether the no-radar gain is stable and whether seed diversity supports a single compressed parameter soup.

## V22 breakthrough and compression status

| Candidate | Fold 0 | Fold 1 | Combined endpoint |
|---|---:|---:|---:|
| V18 no-radar | 72.85% | 80.00% | 1039/1357 |
| Progressive 160 | 73.62% | 79.29% | 1039/1357 |
| Masked-modality distillation | 72.85% | 78.16% | 1026/1357 |
| Seed-43 reproduction | 70.86% | 78.01% | 1012/1357 |

Masked-modality distillation improves visually incomplete rows from `19 -> 23` correct on fold 0 and `62 -> 64` on fold 1, despite its lower total endpoint. A five-candidate convex probability ensemble uses weights fitted only by fold-0 NLL and transfers them unchanged to fold 1. It reaches `487/652` and `574/705`, or `1061/1357 = 78.19%`, 22 samples above V18. Under the frozen domain pair-aware decoder it reaches `1178/1357`.

A second-stage acceptance rule compares this decoded candidate against the prior V14 final prediction. The single `candidate_margin >= 0.4` rule is selected on fold 0, where it preserves `559` correct, then raises blind fold 1 from `627` to `629`. The resulting honest local score is **`1188/1357 = 87.55%`**, two samples above the previous `1186` local best.

The five-model probability ensemble is a research upper bound rather than a valid 100 MB package. A weighted soup of the four architecture-compatible R(2+1)D models quantizes to `65,397,487` bytes and reaches `486/652 + 561/705 = 1047/1357` as one model, but its second-stage system score is only `1184/1357`. V23 therefore saves all epochs of an exact V18 reproduction for a fold-0-selected, fold-1-blind epoch soup, while warm-started masked-distillation and progressive-resolution candidates continue.

## V23 local-best result

Warm-started masked-modality distillation raises the single-model endpoint to `479/652 + 565/705 = 1044/1357`. A parameter soup then combines no-radar, cross-user, progressive-160, scratch masked-distillation, and warm-started masked-distillation checkpoints. Its final source weights are `0.082416/0.079504/0.148508/0.089572/0.600000`, selected only from fold 0 by top-1 accuracy followed by NLL tie-breaking.

The resulting single R(2+1)D model reaches **`487/652 + 567/705 = 1054/1357`**, fifteen samples above V18. Its INT8 checkpoint is `65,397,018` bytes. A direct hybrid INT4 checkpoint is `36,413,345` bytes and retains `480/652 + 564/705 = 1044/1357`.

Under candidate-domain pair-aware decoding, the INT8 branch reaches `1179/1357`. When used as a second-stage challenger to the frozen V14 final system, the fold-0-selected `candidate_agrees` rule scores `566/652` and transfers blind to `624/705` on fold 1. This establishes the new honest local best of **`1190/1357 = 87.69%`**, four samples above V14. The exact report is `experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_second_stage_acceptance.json`.

A probability-space blend can reach `494/652 + 576/705 = 1070/1357`, but it falls to `1185/1357` after sequence decoding. This is stronger evidence that the next major bottleneck is decoder/routing compatibility, not raw clip classification alone.

The complete `1190` system still exceeds 100 MB when the new branch is added naively to the existing package. A rank-16 research factorization against the dequantized fixed-source target is `3,004,806` bytes and reaches `1188/1357`; rank 1 fits the current package directly but reaches only `1180/1357`. Crucially, reconstructing rank 16 against the actual uniformly quantized packaged base falls to `467/652 + 546/705` raw and `1178/1357` as a system, so this path is rejected rather than presented as deployable. The single-branch gain is compressed, while packaging the complete two-stage local best requires removing duplicated assets or distilling its routing behavior.

## V23 public submission check

An exploratory compliant submission averages the two complementary-fold V23 parameter soups into one INT8 checkpoint, then applies the parameter-free pair-aware decoder. The checkpoint plus YOLO assets total `71,006,837` bytes. Every training sample contributes to at least one source fold, but this is parameter averaging rather than a full-data refit and has no honest held-out estimate.

Kaggle submission `56135222` scores **`0.85074`**, below the V13 public best `0.89552`. The result rejects cross-fold parameter averaging as a deployment substitute. V13 remains the public champion; the local `1190/1357` system is not claimed as a public improvement until its routing gain is distilled into a full-data model under 100 MB.
