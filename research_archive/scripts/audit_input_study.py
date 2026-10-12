import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_thermal_motion import motion_box
from src.data import ROOT, ClipDataset, get_indices, get_records, prepare_batch
from src.scratch_models import ScratchMultiModalModel, validate_scratch_config


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    configuration = json.loads((ROOT / "configs/input_study.json").read_text())
    metadata = json.loads((ROOT / "data/metadata/thermal_motion.json").read_text())
    assert metadata["status"] == "complete" and metadata["clips"] == len(get_records())
    assert metadata["metadata_sha256"] == hashlib.sha256((ROOT / "data/metadata/samples.json").read_bytes()).hexdigest()
    assert metadata["source_sha256"] == hashlib.sha256((ROOT / "scripts/prepare_thermal_motion.py").read_bytes()).hexdigest()
    static = np.zeros((24, 3, 160, 160), dtype=np.uint8)
    assert motion_box(static) == [0, 0, 1, 1]
    for step in range(24):
        static[step, :, 50:100, 60 + step:75 + step] = 200
    box = motion_box(static)
    assert box[0] <= 60 / 160 and box[1] <= 50 / 160 and box[2] >= 98 / 160 and box[3] >= 100 / 160
    for record in get_records():
        cache = np.load(ROOT / f"data/processed/thermal_motion_24_160/{record['index']:05d}.npy", mmap_mode="r")
        assert cache.shape == (24, 3, 160, 160) and cache.dtype == np.uint8
    cases = {}
    for modality, options in [("imu", {"imu_representation": "world"}), ("thermal", {"thermal_view": "motion_crop"})]:
        dataset = ClipDataset([0, 1], [modality], frames=4, image_size=160, **options)
        baseline = ClipDataset([0, 1], [modality], frames=4, image_size=160)
        batch = {name: torch.as_tensor(np.stack([dataset[index][name] for index in range(2)])) for name in dataset[0]}
        np.testing.assert_array_equal(batch[modality + "_mask"].numpy(), [baseline[index][modality + "_mask"] for index in range(2)])
        if modality == "imu":
            np.testing.assert_array_equal(dataset[0]["imu"], np.load(ROOT / "data/processed/imu_world.npy", mmap_mode="r")[0])
        batch = prepare_batch(batch, torch.device("cpu"))
        model = ScratchMultiModalModel([modality], **options)
        reference = ScratchMultiModalModel([modality])
        assert {name: value.shape for name, value in model.state_dict().items()} == {name: value.shape for name, value in reference.state_dict().items()}
        logits, auxiliary = model(batch)
        (logits.square().mean() + auxiliary[modality].square().mean()).backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        model.eval()
        state = io.BytesIO()
        torch.save(model.state_dict(), state)
        state.seek(0)
        restored = ScratchMultiModalModel([modality], **options).eval()
        restored.load_state_dict(torch.load(state, weights_only=True))
        torch.testing.assert_close(model(batch)[0], restored(batch)[0], atol=0, rtol=0)
        batch[modality + "_mask"].zero_()
        batch[modality].fill_(float("nan"))
        torch.testing.assert_close(model(batch)[0], model.class_prior_logits.expand(2, -1))
        cases[modality] = {"parameters": sum(parameter.numel() for parameter in model.parameters()), "state_bytes": state.getbuffer().nbytes}
    for job in configuration["jobs"]:
        validate_scratch_config(configuration | job)
        get_indices(get_records(), job.get("fold", configuration["fold"]))
    for invalid in [{"modalities": ["imu"], "thermal_view": "motion_crop"}, {"modalities": ["thermal"], "imu_representation": "world"},
                    {"modalities": ["thermal"], "thermal_view": "motion_crop", "image_size": 96}]:
        try:
            validate_scratch_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input configuration accepted")
    canvas = Image.new("RGB", (640, 510))
    draw = ImageDraw.Draw(canvas)
    for position, index in enumerate([0, 100, 500, 1000, 1600, 2200]):
        full = np.load(ROOT / f"data/processed/video_24_160/{index:05d}.npy", mmap_mode="r")[12, 4:]
        crop = np.load(ROOT / f"data/processed/thermal_motion_24_160/{index:05d}.npy", mmap_mode="r")[12]
        column, row = position % 2 * 320, position // 2 * 170
        canvas.paste(Image.fromarray(full.transpose(1, 2, 0)), (column, row + 10))
        canvas.paste(Image.fromarray(crop.transpose(1, 2, 0)), (column + 160, row + 10))
        draw.text((column, row), f"{index}: full / motion crop", fill="white")
    destination = ROOT / "outputs/study_inputs"
    destination.mkdir(parents=True, exist_ok=True)
    canvas.save(destination / "thermal_crops.png")
    sources = ["src/data.py", "src/scratch_models.py", "src/encoder_initialization.py", "scripts/train.py", "scripts/predict.py",
               "scripts/run_scratch_study.py", "scripts/prepare_thermal_motion.py", "scripts/audit_input_study.py",
               "configs/input_study.json", "data/metadata/thermal_motion.json", "data/processed/imu_world.npy"]
    report = {"status": "passed", "cases": cases, "clips": metadata["clips"],
              "checks": ["synthetic static and moving clips", "all cache shapes", "unchanged coverage and parameters", "finite gradients",
                         "serialization parity", "missing NaN masks", "invalid configurations", "user isolation"],
              "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}}
    (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
