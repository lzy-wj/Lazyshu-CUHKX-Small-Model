#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RELIABILITY = {"Depth_Color": 0.95, "Thermal": 1.0}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-threshold", type=float, choices=(0.1, 0.2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    old = json.loads(
        (ROOT / "data/processed/r2plus1d_yolo_all_visual_32_128/windows.json").read_text()
    )
    ir = json.loads((HERE / "outputs/IR_windows.json").read_text())
    alternatives = {
        modality: json.loads((HERE / f"outputs/{modality}_windows.json").read_text())
        for modality in RELIABILITY
    }
    windows = list(old)
    changed = full_frame = 0
    selected = {modality: 0 for modality in RELIABILITY}
    for index in range(len(windows)):
        if ir["quality"][index] * 0.90 >= args.ir_threshold:
            continue
        candidates = []
        for modality, values in alternatives.items():
            window = values["windows"][index]
            quality = values["quality"][index] * RELIABILITY[modality]
            if window is not None and quality >= 0.10:
                candidates.append((quality, modality, window))
        if candidates:
            _, modality, window = max(candidates)
            windows[index] = window
            selected[modality] += 1
        else:
            windows[index] = None
            full_frame += 1
        changed += windows[index] != old[index]
    args.output.write_text(json.dumps(windows) + "\n")
    report = {
        "ir_threshold": args.ir_threshold,
        "changed_windows": changed,
        "alternative_modalities": selected,
        "full_frame_fallbacks": full_frame,
        "output": str(args.output),
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

