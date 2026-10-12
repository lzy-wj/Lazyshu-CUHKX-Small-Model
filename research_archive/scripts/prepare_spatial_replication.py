import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_spatial_study import write_frozen
from src.data import ROOT


def main():
    source = ROOT / "configs/spatial_token_study.json"
    audit_path = ROOT / "outputs/study_spatial/audit.json"
    config = json.loads(source.read_text())
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "passed"
    for name, digest in audit["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    config["name"] = "spatial_tokens_v1_s43"
    config["seed"] = 43
    config["jobs"] = [job | {"name": job["name"].replace("spatial_v1_", "spatial_v1_s43_")} for job in config["jobs"]]
    config["replication"] = "Fixed seed43 repeat of all three architectures on both identical grouped splits; no architecture omitted based on seed42 results."
    config["queue_hours"] = 4
    destination = ROOT / "configs/spatial_token_study_s43.json"
    write_frozen(destination, config)
    replica_audit = copy.deepcopy(audit)
    replica_audit["scope"] = "Same audited architecture, data and training code; only seed and unique run names differ."
    replica_audit["source_sha256"][str(destination.relative_to(ROOT))] = hashlib.sha256(destination.read_bytes()).hexdigest()
    replica_audit["source_sha256"][str(audit_path.relative_to(ROOT))] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    folder = ROOT / "outputs/study_spatial_s43"
    folder.mkdir(parents=True, exist_ok=True)
    write_frozen(folder / "audit.json", replica_audit)
    print("Frozen seed43 replication of all six jobs")


if __name__ == "__main__":
    main()
