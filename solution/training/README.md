# Training code and lineage

`source_snapshot/` preserves the participant source used to create the frozen base, adapters, residual modules, and structural decoders. Generated checkpoints, predictions, logs, SwanLab runs, caches, Kaggle probe CSVs, and CUHK-X raw data are deliberately excluded. Frozen base-run configs are in `frozen_configs/` and the experiment protocol is in `docs/`.

## Data layout

From `training/source_snapshot/`, place the organizer-authorized dataset under `data/raw/` exactly as used by `scripts/prepare_data.py`, and place the published Small-Model-Track Testing CSV at:

```text
data/downloads/CUHK-X_Small_Model_Track/Small-Model-Track/Testing/test_file/test.csv
```

The package does not redistribute any CUHK-X data. Building the final full-data model requires the original labeled Training set; inference does not.

## Training-only initialization

Download the exact R(2+1)D initialization during environment setup, not inference:

```bash
mkdir -p .tools/checkpoints
curl -L -o .tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth \
  https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth
echo 'ade133f18548b59da3a36d6a70d44a4a7cc4b90460654c338cb8a023e9a3e0e1  .tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth' | sha256sum -c -
cp ../../../artifacts/yolo11n_compact.pt .tools/checkpoints/yolo11n.pt
```

## Base preprocessing and five-source full fit

```bash
python scripts/prepare_data.py --stage index
python scripts/prepare_data.py --stage sensors --workers 16
python scripts/prepare_imu_world.py
python scripts/prepare_r2plus1d_cache.py \
  --output data/processed/r2plus1d_yolo_all_visual_32_128 \
  --yolo .tools/checkpoints/yolo11n.pt --frames 32 --image-size 128 \
  --rgb-modalities Depth_Color,Thermal
python scripts/prepare_r2plus1d_cache.py \
  --output data/processed/r2plus1d_yolo_all_visual_32_160 \
  --yolo .tools/checkpoints/yolo11n.pt --frames 32 --image-size 160 \
  --rgb-modalities Depth_Color,Thermal \
  --windows-from data/processed/r2plus1d_yolo_all_visual_32_128/windows.json

export PYTHON="$(command -v python)"
export SWANLAB_MODE=disabled
experiments/v24_full_refit/run_full_no_radar.sh 0
experiments/v24_full_refit/run_full_cross_user.sh 1
experiments/v24_full_refit/run_full_scratch_maskdistill.sh 2
experiments/v24_full_refit/run_full_progressive160.sh 3
experiments/v24_full_refit/run_full_warm_maskdistill.sh 4
```

The fixed epochs, learning rates, augmentations, source hashes, and soup weights are recorded in `experiments/v24_full_refit/README.md`, `protocol.json`, the five JSON files in `frozen_configs/`, and the checkpoint metadata itself. `scripts/soup_r2plus1d_checkpoints.py` and the V24 finalizer form and quantize the weighted parameter soup.

## Post-base stages

The exact post-base source lineage is retained in these directories:

- `scene_context_v13`: test-pseudo head adaptation and pair-aware decoders;
- `v52_domain_logit_residual`, `v57_round_transformer_distillation`, `v70_film_margin_distillation`, `v71_gated_film_distillation`: compact residual distillation;
- `v87_v24_highres_transfer`, `v89_pseudo_multiscale`: high-resolution pseudo adaptation;
- `v96_modality_specialized_backbone`, `v97_protected_visual_adapters`, `v98_cross_modal_target_adapter`, `v103_full_protected_adapters`, `v104_domain_gated_adapters`: protected visual and cross-modal adapters;
- `v109_crossfit_round_transformer`, `v113_paired_person_adapters`, `v114_paired_person_fusion`: sequence and paired-person adapters;
- `v125_domain_complement`, `v127_supervision_routed_system`: supervision-aware final routing and public date ablation;
- `v160_radar_batch_decoder`, `v164_evidence_gated_radar`: Radar batch grouping and evidence gate.

These stages require the generated outputs of their predecessors. Their run scripts and checkpoint-embedded configs record every training hyperparameter. `artifacts/decoder_state.json` is the frozen export of label-derived training transitions/templates plus routing constants; it is loaded directly at inference so Training labels are never required by `inference.sh`.

## Experiment tracking

Training originally used `swanlab==0.8.4` in cloud mode. It is optional for reconstruction: set `SWANLAB_MODE=disabled` as above. Install training-only review dependencies with `python -m pip install -r training/requirements.txt` from the package root.
