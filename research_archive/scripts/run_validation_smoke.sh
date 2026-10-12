#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash scripts/run_training.sh "${1:-7}" validation_smoke_concat --strict-scratch --canonical-pose \
    --diagnostic-split configs/same_user_diagnostic_split.json \
    --modalities scene_ir,thermal,scene_depth,skeleton,imu,radar --fusion concat \
    --epochs 1 --batch-size 12 --frames 4 --image-size 96 --workers 2 --limit-batches 2 --swanlab-mode disabled
conda run --no-capture-output -n Train python -c '
import json
from pathlib import Path
import numpy as np
folder=Path("outputs/checkpoints/validation_smoke_concat")
split=json.loads((folder/"split.json").read_text())
config=json.loads((folder/"config.json").read_text())
assert [len(split[key]) for key in ["train","valid","outer_valid"]] == [1907,477,652]
assert config["selection_partition"] == "same_user_holdout"
assert not config["encoder_sources"] and not config["pretrained"]
best=json.loads((folder/"diagnostic_best.json").read_text())
metrics=json.loads((folder/"metrics.json").read_text())
assert metrics["accuracy"] == best["inner_valid"]["accuracy"]
for filename, partition in [("clean_train.npz","train"),("validation.npz","valid"),("outer_valid.npz","outer_valid")]:
    with np.load(folder/filename) as values:
        np.testing.assert_array_equal(values["indices"], split[partition])
print("Nested validation smoke passed: disjoint partitions, fresh initialization, complete predictions, inner-only selection")
'
