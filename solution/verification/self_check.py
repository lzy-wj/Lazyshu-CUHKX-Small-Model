#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-unsigned", action="store_true")
    args = parser.parse_args()

    hashes = json.loads((ROOT / "configs/artifact_hashes.json").read_text())
    total = 0
    for relative, expected in hashes["files"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size == expected["bytes"], relative
        assert sha256(path) == expected["sha256"], relative
        total += path.stat().st_size
    assert total == hashes["total_bytes"] == 84_094_248
    assert total < hashes["competition_limit_bytes"] == 100_000_000

    submissions = yaml.safe_load((ROOT / "submission/submission_info.yaml").read_text())
    assert submissions["primary_verification_submission_id"] == "56174717"
    for item in submissions["selected_final_submissions"]:
        assert sha256(ROOT / item["csv_path"]) == item["csv_sha256"]

    manifest = yaml.safe_load((ROOT / "manifest.yaml").read_text())
    assert manifest["team"]["name"] == "Lazyshu"
    assert manifest["model_budget"]["bundled_model_asset_bytes"] == total
    if not args.allow_unsigned:
        assert manifest["team"]["contact_email"]
        signed = manifest["declarations"]["signed_declaration"]
        assert signed and (ROOT / signed).is_file()

    forbidden = ("REPLACE" + "_ME", "TO_BE" + "_PROVIDED")
    participant_home = "/data/home/" + "lzy"
    text_suffixes = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".txt"}
    for path in ROOT.rglob("*"):
        if path.is_dir():
            assert path.name not in {"__pycache__", ".lazyshu_inference_cache"}
            continue
        if path.suffix in text_suffixes:
            text = path.read_text(errors="replace")
            for token in forbidden:
                assert token not in text, f"{token}: {path.relative_to(ROOT)}"
            relative = path.relative_to(ROOT)
            archival_training_file = relative.parts[:2] in {
                ("training", "source_snapshot"),
                ("training", "frozen_configs"),
            }
            if not archival_training_file:
                assert participant_home not in text, f"absolute path: {relative}"
    print(json.dumps({"status": "ok", "model_asset_bytes": total}, indent=2))


if __name__ == "__main__":
    main()
