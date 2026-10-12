import io
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset
from src.scratch_models import ScratchMultiModalModel, skeleton_streams, validate_scratch_config


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    values = torch.stack([torch.from_numpy(ClipDataset([index], ["skeleton"], canonical_pose=True)[0]["skeleton"])
                          for index in [0, 1, 2]])
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    transformed = values.clone()
    pose = transformed[:, :, :51].reshape(3, 64, 17, 3)
    pose[:] = pose @ rotation * 1.7 + torch.tensor([2.0, -3.0, 4.0])
    for original, altered in zip(skeleton_streams(values, "invariant"), skeleton_streams(transformed, "invariant")):
        torch.testing.assert_close(original, altered, atol=3e-5, rtol=3e-5)
    centered = skeleton_streams(values, "centered")
    assert centered[0].reshape(3, 64, 17, 3)[:, :, 0].abs().max() == 0
    assert centered[2].reshape(3, 64, 17, 3)[:, :, 0].abs().max() == 0
    results = {}
    for representation in ["coordinates", "centered", "invariant"]:
        model = ScratchMultiModalModel(["skeleton"], skeleton_representation=representation)
        batch = {"skeleton": values.clone(), "skeleton_mask": torch.tensor([1.0, 1.0, 0.0])}
        batch["skeleton"][2] = float("nan")
        logits, auxiliary = model(batch)
        loss = logits[:2].square().mean() + auxiliary["skeleton"][:2].square().mean()
        loss.backward()
        assert torch.isfinite(logits).all()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        assert all(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in stream.parameters())
                   for stream in model.encoders["skeleton"].streams)
        torch.testing.assert_close(logits[2], model.class_prior_logits)
        model.eval()
        state = io.BytesIO()
        torch.save(model.state_dict(), state)
        state.seek(0)
        restored = ScratchMultiModalModel(["skeleton"], skeleton_representation=representation).eval()
        restored.load_state_dict(torch.load(state, weights_only=True))
        torch.testing.assert_close(model(batch)[0], restored(batch)[0], atol=0, rtol=0)
        results[representation] = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                                   "state_bytes": state.getbuffer().nbytes}
    for config in [{"modalities": ["skeleton"], "skeleton_representation": "unknown"},
                   {"modalities": ["imu"], "skeleton_representation": "centered"}]:
        try:
            validate_scratch_config(config)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid representation configuration accepted")
    checkpoint = torch.load(ROOT / "outputs/checkpoints/validation_v1_skeleton_f0/best.pt", map_location="cpu", weights_only=False)
    ScratchMultiModalModel(["skeleton"]).load_state_dict(checkpoint["model"], strict=True)
    report = {"status": "passed", "checks": ["rigid/scale invariance", "centered root", "finite gradients", "missing NaN masking",
                                               "serialization parity", "invalid configuration rejection", "old checkpoint compatibility"],
              "models": results}
    destination = ROOT / "outputs/study_representations"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
