import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT
from src.scratch_augmentation import single_modality_mixup
from src.scratch_models import ScratchMultiModalModel, validate_scratch_config


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    batch = {"skeleton": torch.randn(4, 64, 153), "skeleton_mask": torch.ones(4),
             "label": torch.arange(4), "index": torch.arange(4)}
    permutation = torch.tensor([1, 0, 3, 2])
    for coefficient in [0, 0.2, 1]:
        mixed = single_modality_mixup(batch, "skeleton", coefficient, permutation)
        torch.testing.assert_close(mixed["skeleton"], coefficient * batch["skeleton"] + (1 - coefficient) * batch["skeleton"][permutation])
        for name in ["skeleton_mask", "label", "index"]:
            torch.testing.assert_close(mixed[name], batch[name], atol=0, rtol=0)
    model = ScratchMultiModalModel(["skeleton"])
    mixed = single_modality_mixup(batch, "skeleton", 0.2, permutation)
    logits, auxiliary = model(mixed)
    loss = 0.2 * torch.nn.functional.cross_entropy(logits, batch["label"])
    loss += 0.8 * torch.nn.functional.cross_entropy(logits, batch["label"][permutation])
    loss += auxiliary["skeleton"].square().mean()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for broken in [dict(batch, skeleton_mask=torch.tensor([1, 1, 0, 1])), dict(batch, imu_mask=torch.ones(4))]:
        try:
            single_modality_mixup(broken, "skeleton", 0.2, permutation)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe missing-mask mixing accepted")
    configuration = json.loads((ROOT / "configs/skeleton_regularization.json").read_text())
    validate_scratch_config(configuration)
    for broken in [dict(configuration, modalities=["skeleton", "imu"]), dict(configuration, mixup=0),
                   dict(configuration, single_modality_mixup=False)]:
        try:
            validate_scratch_config(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid mixup configuration accepted")
    print("Passed: feature mixing, endpoints, unchanged masks/labels/indices, gradients, missing-mask rejection, strict configuration guards")


if __name__ == "__main__":
    main()
