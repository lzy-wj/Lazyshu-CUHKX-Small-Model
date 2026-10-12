# Tested environment

- OS: Ubuntu 22.04 LTS, Linux x86_64
- Python: 3.12.12
- NVIDIA driver: 580.126.20
- CUDA packaged with PyTorch: 12.8
- PyTorch: 2.9.1+cu128
- TorchVision: 0.24.1+cu128
- GPU: NVIDIA A800-SXM4-80GB; code requires an Ampere-or-newer CUDA GPU with BF16 support
- System packages: no compiler, FFmpeg, or custom CUDA extension is required

Create the environment from the package root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.1.2
python -m pip install -r environment/requirements.txt
python - <<'PY'
import torch, torchvision, ultralytics
print(torch.__version__, torch.version.cuda, torchvision.__version__, ultralytics.__version__)
assert torch.cuda.is_available()
PY
```

Package installation needs network access to PyPI and the PyTorch wheel index. Runtime inference is offline and all weights are included. Third-party licences are summarized in `../THIRD_PARTY_NOTICES.md`; the bundled Ultralytics AGPL-3.0 text is in `../licenses/`.
