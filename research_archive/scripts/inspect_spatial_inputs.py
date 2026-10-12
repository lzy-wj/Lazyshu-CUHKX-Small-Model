import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records


def main():
    records = get_records()
    fit = json.loads((ROOT / "configs/spatial_grouped_f0.json").read_text())["train"]
    labels = {6: "Drink water", 7: "Eat food", 12: "Sweep", 13: "Mop", 21: "Read", 22: "Turn pages"}
    selected = []
    for label in labels:
        candidates = [records[index] for index in fit if records[index]["label"] == label
                      and all(name in records[index]["modalities"] for name in ["IR", "Depth_Color", "Thermal"])]
        candidates.sort(key=lambda record: (record["user"] != 1, record["index"]))
        selected.append(candidates[0])
    canvas = Image.new("RGB", (1100, len(selected) * 205 + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for column, heading in enumerate(["Raw IR middle frame", "Raw Depth middle frame", "Raw Thermal middle frame", "Thermal crop input", "Thermal + 16px patch grid"]):
        draw.text((column * 220 + 4, 8), heading, fill="black")
    manifest = []
    for row, record in enumerate(selected):
        images = []
        for name in ["IR", "Depth_Color", "Thermal"]:
            paths = sorted(path for path in (ROOT / record["modalities"][name]).rglob("*") if path.suffix.lower() in [".png", ".jpg", ".jpeg"])
            image = Image.open(paths[len(paths) // 2]).convert("RGB")
            images.append(image)
        values = np.load(ROOT / f"data/processed/thermal_motion_24_160/{record['index']:05d}.npy", mmap_mode="r")
        crop = Image.fromarray(np.asarray(values[len(values) // 2]).transpose(1, 2, 0).astype(np.uint8))
        images.append(crop)
        grid = crop.copy()
        grid_draw = ImageDraw.Draw(grid)
        for position in range(16, 160, 16):
            grid_draw.line((position, 0, position, 159), fill="white")
            grid_draw.line((0, position, 159, position), fill="white")
        images.append(grid)
        for column, image in enumerate(images):
            image.thumbnail((212, 160))
            top = 32 + row * 205
            canvas.paste(image, (column * 220 + 4, top))
            draw.text((column * 220 + 4, top + 164), f"{labels[record['label']]} | user {record['user']} | idx {record['index']}", fill="black")
        manifest.append({"index": record["index"], "label": record["label"], "user": record["user"], "partition": "fold0_fit"})
    folder = ROOT / "outputs/model_analysis"
    canvas.save(folder / "spatial_input_contact.png")
    (folder / "spatial_input_contact.json").write_text(json.dumps({"samples": manifest,
        "caveat": "Six training-only examples; independently chosen middle frames are not calibrated cross-camera correspondences. Display thumbnails do not measure raw resolution."}, indent=2) + "\n")
    print(folder / "spatial_input_contact.png")


if __name__ == "__main__":
    main()
