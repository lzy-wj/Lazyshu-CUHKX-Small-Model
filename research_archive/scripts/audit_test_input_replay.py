import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_data import sensor_clip
from scripts.prepare_imu_world import features as world_features
from src.data import ROOT, get_records


def main():
    records = [record for record in get_records() if record["split"] == "test"]
    metadata = json.loads((ROOT / "data/metadata/thermal_motion.json").read_text())
    details = {entry["index"]: entry for entry in metadata["clips_detail"]}
    skeleton = np.load(ROOT / "data/processed/skeleton.npy", mmap_mode="r")
    imu = np.load(ROOT / "data/processed/imu_world.npy", mmap_mode="r")
    results = []
    for position in np.linspace(0, len(records) - 1, 20).round().astype(int):
        record = records[position]
        assert record["label"] == -1
        index = record["index"]
        fresh_skeleton, _, _, _ = sensor_clip(record)
        np.testing.assert_array_equal(fresh_skeleton, skeleton[index])
        np.testing.assert_array_equal(world_features(record), imu[index])
        detail = details[index]
        checked = 0
        if "Thermal" in record["modalities"] and not detail["errors"]:
            files = sorted(path for path in (ROOT / record["modalities"]["Thermal"]).iterdir()
                           if path.suffix.lower() in [".png", ".jpg", ".jpeg"] and not path.name.startswith("."))
            if files:
                selected = np.linspace(0, len(files) - 1, 24).round().astype(int)
                cached = np.load(ROOT / f"data/processed/thermal_motion_24_160/{index:05d}.npy", mmap_mode="r")
                box = detail["box"]
                for step in [0, 12, 23]:
                    with Image.open(files[selected[step]]) as image:
                        image = image.convert("RGB")
                        width, height = image.size
                        pixels = (int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height))
                        fresh = np.asarray(image.crop(pixels).resize((160, 160), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
                        np.testing.assert_array_equal(fresh, cached[step])
                        checked += 1
        results.append({"index": index, "skeleton_exact": True, "imu_world_exact": True, "thermal_frames_exact": checked})
    report = {"status": "passed", "test_clips_replayed": len(results), "results": results,
              "scope": "Read-only raw-to-cache replay on 20 evenly spaced unlabeled test clips; no overwrites or label-based decisions"}
    (ROOT / "outputs/study_inputs/test_input_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Raw-to-cache replay passed:", len(results), "test clips;", sum(result["thermal_frames_exact"] for result in results), "thermal frames")


if __name__ == "__main__":
    main()
