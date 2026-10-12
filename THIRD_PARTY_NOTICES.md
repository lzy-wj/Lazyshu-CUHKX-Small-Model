# Third-party models and software

## R(2+1)D-34 initialization

- Purpose: initialization of the visual CNN before full CUHK-X supervised training; not downloaded or separately loaded during inference because it is incorporated into the bundled final checkpoint.
- Model: `r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth`, IG-65M pretraining followed by Kinetics-400 fine-tuning.
- Source: `https://github.com/moabitcoin/ig65m-pytorch/releases/tag/v1.0.0`
- Exact weight URL: `https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth`
- SHA-256: `ade133f18548b59da3a36d6a70d44a4a7cc4b90460654c338cb8a023e9a3e0e1`
- Repository licence: MIT. The underlying IG-65M and Kinetics datasets retain their original terms; the team used only the public pretrained checkpoint.

## Ultralytics YOLO11n

- Purpose: deterministic person detection for visual crop-window construction; never used to produce activity classes.
- Source: `https://github.com/ultralytics/ultralytics`, YOLO11 documentation `https://docs.ultralytics.com/models/yolo11/`
- Runtime software version: `ultralytics==8.4.142`
- Training data: COCO object detection dataset through the public pretrained checkpoint.
- Bundled checkpoint: `artifacts/yolo11n_compact.pt`, SHA-256 `11120a721121fb9d178b87cb769027f848b3dfae138db8ff08db3b6920389999`
- Licence: GNU Affero General Public License v3.0; text included as `licenses/ULTRALYTICS_AGPL-3.0.txt`.

## Runtime libraries

PyTorch, TorchVision, NumPy, pandas, SciPy, Pillow, OpenCV, tqdm, and their declared dependencies retain their respective upstream licences. Exact versions are pinned in `environment/requirements.txt`. No closed-source API or paid service is used.
