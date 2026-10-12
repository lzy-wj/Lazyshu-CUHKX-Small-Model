import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import evaluate
from src.data import ROOT, ClipDataset, get_records, prepare_batch
from src.scratch_models import ScratchMultiModalModel
from src.validation_splits import load_diagnostic_split


def main():
    torch.set_num_threads(4)
    fit, inner, outer = load_diagnostic_split(ROOT / "configs/same_user_diagnostic_split.json", get_records(), 0)
    modalities = ["scene_ir", "thermal", "scene_depth", "skeleton", "imu", "radar"]
    model = ScratchMultiModalModel(modalities)
    dataset = ClipDataset(fit[:2], modalities, frames=4, image_size=96, canonical_pose=True)
    batch = prepare_batch(next(iter(DataLoader(dataset, batch_size=2))), torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    torch.nn.functional.cross_entropy(model(batch)[0], batch["label"]).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    optimizer.step()
    model.eval()
    before = {name: values.clone() for name, values in model.state_dict().items()}
    for indices in [fit[:2], inner[:2], outer[:2]]:
        dataset = ClipDataset(indices, modalities, frames=4, image_size=96, canonical_pose=True)
        metrics, predictions = evaluate(model, DataLoader(dataset, batch_size=2), torch.device("cpu"))
        np.testing.assert_array_equal(predictions["indices"], indices)
        assert np.isfinite(predictions["probabilities"]).all() and np.isfinite(metrics["nll"])
    for name, values in before.items():
        torch.testing.assert_close(model.state_dict()[name], values, rtol=0, atol=0)
    rejected = []
    for forbidden in [["--init", "outputs/checkpoints/scratch_v1_concat_f0/best.pt"],
                      ["--encoder-sources", "outputs/checkpoints/scratch_v1_skeleton_f0/best.pt"]]:
        command = [sys.executable, "scripts/train.py", "--name", "invalid_nested_init", "--strict-scratch",
                   "--diagnostic-split", "configs/same_user_diagnostic_split.json", *forbidden]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode != 0 and "ValueError" in result.stderr
        assert not (ROOT / "outputs/checkpoints/invalid_nested_init").exists()
        rejected.append(forbidden[0])
    paths = ["scripts/train.py", "scripts/run_training.sh", "scripts/run_validation_smoke.sh", "scripts/run_validation_queue.py",
             "src/data.py", "src/scratch_models.py", "src/validation_splits.py", "configs/same_user_diagnostic_split.json"]
    report = {"status": "passed", "counts": [len(fit), len(inner), len(outer)], "rejected_initializations": rejected,
              "scope": "CPU real-data forward/backward, three clean evaluations without model mutation, partition indices, and forbidden reuse of old weights",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}}
    (ROOT / "outputs/study_validation/cpu_pipeline_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
