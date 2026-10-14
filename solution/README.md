# Team Lazyshu Public Source Guide

This directory contains the final inference source, training snapshot, configurations, and internal verification records for Team **Lazyshu** in the CUHK-X Small Model Track. The primary submission `56174717` (V127) achieved private accuracy `0.91176` and fourth place on the Kaggle private leaderboard. The team advanced to the UbiComp 2026 finals and received an Outstanding Award.

**This public checkout does not include the 14 runtime artifacts, decoder state, prediction CSVs, submission metadata, or signed declarations.** The original private frozen package reproduced both selected submissions. The environment, commands, and measurements below describe that package; the public checkout alone cannot run final inference. See [the release boundary](../docs/REPRODUCIBILITY.md).

## 0 Needs attention

These requirements apply after the exact frozen runtime artifacts have been restored.

- More than one GPU needed: **no**
- More than 24 GB of GPU memory needed: **no**; measured peak allocated memory was **13.49 GB**
- Specific GPU generation or driver needed: **yes**; NVIDIA Ampere or newer with BF16 support and a CUDA 12.8-compatible driver. Tested with driver `580.126.20`.
- Network access needed at inference time: **no**
- Runs on Linux x86_64: **yes**

## 1 Team and submission

- Team name: `Lazyshu`
- Track: `small`
- Selected Kaggle Final Submission record: `submission/submission_info.yaml` in the private frozen package; excluded from this public release
- Primary: ID `56174717`, `v127_without_0531.csv`, submitted `2026-09-12T00:11:01.613Z`, public `0.92039`, private `0.91176`, SHA-256 `76210e13f1f63a3d63c356f2462aa09917a6d9c0c919a0e8025098ab4408acb0`
- Secondary: ID `56243275`, `v164_evidence_gated_radar.csv`, submitted `2026-09-15T02:30:18.600Z`, public `0.91044`, private `0.90686`, SHA-256 `aaf3ac9dc0fe71fe5a21b78d2243ac330dee3bec0aad69ebb52f11acb7989639`
- Primary verification target: `56174717`
- Solution type: compact multimodal ensemble with local checkpoints; no API or remote service

The exact pre-upload CSV bytes are retained in the private package. They were not opened, resaved, reordered, or normalized while constructing that package. The public `submission/` directory contains only a release-boundary note.

## 2 Frozen solution summary

### Architecture and modalities

The 64,953,671-parameter base classifier is a permitted CNN/RNN/Transformer architecture:

1. A shared R(2+1)D-34 CNN encodes 32 synchronized frames from `Depth_Color`, `Thermal`, and grayscale `IR` as seven channels. It was initialized from the public IG-65M-to-Kinetics-400 R(2+1)D checkpoint and then fully trained on CUHK-X.
2. A temporal CNN encodes 17-joint skeleton coordinates, bones, and velocities. A ConvLSTM-style temporal encoder processes world-oriented IMU features from up to five devices.
3. A two-layer Transformer fuses visual, skeleton, and IMU tokens with explicit missing-modality masks. Raw Radar points are not fed to the base neural network; Radar timestamps are used only by V164's structural grouping rule.
4. The base deployment state is an INT8 weighted parameter soup of five full-data fits. Compact CNN adapters, two learned logit/sequence residual modules, one gated Transformer residual module, and fixed training-derived transition/template statistics are applied afterward.

YOLO11n, pretrained on COCO and stored as `artifacts/yolo11n_compact.pt` in the private package, supplies deterministic person windows only. It never predicts an activity label. Its weights are not included in this public checkout. The organizer explicitly confirmed that this YOLO11n cropper and the IG-65M/Kinetics R(2+1)D initialization are permitted; the archived clarification is `training/docs/OFFICIAL_RULE_CLARIFICATIONS_20260910.md`.

### Training

The frozen V24 base used all 3,036 labeled training clips. Five source runs used fixed 13/16/7/16/9-epoch schedules and soup weights `0.082416/0.079504/0.148508/0.089572/0.600000`. Training used BF16 autocast, FP32 optimization and EMA, clip-consistent horizontal flips, MixUp, and masked-modality distillation in two sources. Seeds were `43` in the full-data launcher (effective trainer seed `42` because the full-data fold is `-1`) and `42` for adapters.

The low/high-resolution head deltas and target adapters were trained only from model-generated pseudo-labels on the official unlabeled Testing set. No test ground truth or manual test annotation was used. This transductive step was explicitly allowed by the organizer and is disclosed because a fixed test-specific adapter may generalize differently on new subjects. Complete training/research source, frozen run configs, and a lineage guide are under `training/`.

### Inference and the two selected variants

Preprocessing detects a union person window, samples 32 frames, builds both 128- and 160-pixel caches, interpolates skeleton/IMU streams to 64 steps, and records modality availability. Inference runs the base, low/high-resolution head adaptations, protected visual adapters, V52/V57/V71 residual modules, BF16 test-time augmentation, and adaptive BatchNorm. Fixed transition/template statistics then perform repeat pooling and pair-aware assignment.

- **V127 / `56174717`:** V71 handles dates without paired source supervision; V114 handles dates with a matched labeled source user. Unsupported source labels are rolled back to V24. The final selected CSV also rolls back changes on acquisition date `2025-05-31`, an ablation chosen from public-leaderboard feedback without access to any test label.
- **V164 / `56243275`:** starts from V127 and accepts a Radar-timestamp decoder challenger only when the original multimodal probability ranks support the challenger and reject the incumbent.

For an unseen acquisition date, the code does not fail or invent a source pairing: it automatically uses the unmatched V71 route. Sample count, file list, and input root are discovered dynamically.

## 3 Hardware and runtime

- Tested operating system: Ubuntu 22.04 LTS, Linux x86_64
- Tested GPU: one NVIDIA A800-SXM4-80GB; minimum is one Ampere-or-newer NVIDIA GPU with 24 GB memory
- Tested driver / runtime: NVIDIA `580.126.20`, PyTorch CUDA `12.8`
- Minimum host memory: 32 GB recommended
- Minimum free disk: 10 GB for the package, environment, and generated 128/160 caches
- Environment setup: approximately 10-20 minutes with package indexes reachable
- Original 405-row Testing inference: **407.77 seconds** total on one A800 (76.21 seconds preprocessing, 311.17 seconds model inference, remainder decoding/I/O)
- Peak allocated CUDA memory: **13.491 GB**
- Inference model assets: **84,094,248 bytes**, below the `100,000,000`-byte limit by `15,905,752` bytes

Runtime scales approximately linearly with clip count. Preprocessing creates about 3.6 GB of cache for the published 405 clips.

## 4 Environment setup

The original package used the accepted pinned-environment option instead of Docker. From the repository root on a clean Linux machine with a CUDA 12.8-compatible NVIDIA driver:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.1.2
python -m pip install -r solution/environment/requirements.txt
```

No compiler or custom CUDA extension is required. Setup needs HTTPS access to `pypi.org`, `files.pythonhosted.org`, and `download.pytorch.org`; inference needs no network after runtime artifacts are restored. Installing dependencies does not supply the missing model assets. The exact tested versions and third-party licence notes are in `environment/README.md` and `THIRD_PARTY_NOTICES.md`.

## 5 Run inference

First restore all 14 assets under `solution/artifacts/` with the exact sizes and hashes in `solution/configs/artifact_hashes.json`. The public repository provides no artifact download. Then run from the repository root:

```bash
# Primary V127 / Submission 56174717
bash solution/inference.sh <data_dir> <output_csv>

# Explicit selected submission
bash solution/inference.sh <data_dir> <output_csv> 56174717
bash solution/inference.sh <data_dir> <output_csv> 56243275
```

Alternatively, change into `solution/` and use `bash inference.sh` with the same arguments. Without the exact artifacts, these commands fail during artifact verification. Independently retrained checkpoints require their own exported decoder state and inventory; the frozen hashes are not compatible with arbitrary replacement weights.

`<data_dir>` may be the published `small_model_track_test` directory itself or a parent preserving the published `Testing/test_file/test.csv` and `small_model_track_test/SM_test_*` layout. Modalities may be missing per clip. The script discovers all clips, preserves `test.csv` order when present, and writes exactly two columns, `path,prediction`, to `<output_csv>`.

Temporary files are written beside the output under `.lazyshu_inference_cache/`; the input tree is read-only. A JSON timing/report file and compressed intermediate probabilities are retained there. Artifact size and SHA-256 checks run before preprocessing; missing data, corrupt artifacts, unsupported Submission IDs, and schema/order mismatches terminate with a nonzero exit status.

No Training labels, test labels, participant-specific absolute path, Hugging Face cache, API, or network resource is accessed at inference time.

## 6 Model artifacts and external resources

The private frozen package contains all 14 runtime assets under `artifacts/`, enumerated with byte counts and SHA-256 values in `configs/artifact_hashes.json` and `manifest.yaml`, and verified before every run. None of those runtime assets is distributed in this public checkout. No runtime download occurs.

The manifest and hash inventory are retained records of the original private package. References to submission metadata, declarations, and artifact files describe that package, not files included in this public release.

External training initialization:

- R(2+1)D-34 clip-32, IG-65M then Kinetics-400, release `v1.0.0`, SHA-256 `ade133f18548b59da3a36d6a70d44a4a7cc4b90460654c338cb8a023e9a3e0e1`, from `moabitcoin/ig65m-pytorch` (MIT repository).
- Ultralytics YOLO11n COCO detector, used only for crop windows. Private compact-weight SHA-256 `11120a721121fb9d178b87cb769027f848b3dfae138db8ff08db3b6920389999`; runtime package version `8.4.142`; AGPL-3.0. The weight is not publicly distributed here.

`THIRD_PARTY_NOTICES.md` provides exact URLs, purposes, and licences. Participant-authored source is supplied under Apache-2.0; third-party components retain their own licences.

## 7 Reproduction expectations

The private package's inference has no stochastic sampling. The stored base weights are INT8, neural compute uses BF16 autocast, and postprocessing uses FP32/FP64 NumPy operations. In an internal audit on the tested A800/PyTorch stack, the entry point reproduced the privately retained `submission/final_submission_1.csv` byte-for-byte with the same SHA-256. Every final label for both selected variants also matched when composed from the fresh model forward pass. These records are not a claim that the public checkout can reproduce those outputs or an organizer certification.

Different BF16 hardware or library kernels can create tiny logit changes. The expected comparison is exact `path` order and exact integer predictions; if a hardware-specific boundary case appears, compare intermediate `probabilities.npz` and use the tested versions before changing any model or method.

## 8 Known limitations and troubleshooting

- The first run spends about 76 seconds building caches; subsequent runs may reuse them only when a content fingerprint matches.
- An out-of-memory error usually means another process occupies the GPU. Use a free GPU or lower only the runtime batch size in `src/predict.py`; this is an environment repair, not a model change.
- Ultralytics may emit update/config warnings in some home directories. In the original private package, the local checkpoint prevented model downloads; the public checkout does not include it.
- The V127 date route and fixed test-pseudo adapters were frozen before the deadline. Unknown dates safely fall back to V71, but their accuracy may differ from the original private set.
- Do not alter artifact files, decoder statistics, TTA, date routing, or adapter weights during verification.

## 9 Public release contents

- Inference source, training snapshot, environment pins, configurations, and historical artifact hashes are included.
- Internal verification reports record the original private package's byte-exact reproduction.
- Runtime weights, decoder state, final CSVs, submission metadata, signed PDF/DOCX declarations, raw challenge data, and inference caches are excluded.
- The signed declaration dated `2026-09-17` UTC is retained privately with the competition package.
- `tools/validate_repository.py` at the repository root checks public-release contents; it does not verify model inference or certify the organizer's open-source requirements.
