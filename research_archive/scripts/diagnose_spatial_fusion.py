import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset, get_records, prepare_batch
from src.scratch_models import ScratchMultiModalModel


def main():
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    records = get_records()
    report = {"partition": "inner validation only", "seed": 42, "experiments": {},
              "caveat": "Inference masking causes distribution shift, not retrained modality ablation. Attention mass is not causal importance. Attention tracing requests explicit weights and can change floating-point execution."}
    for fold in [0, 1]:
        for kind in ["cnn_pooled", "cnn_tokens", "patch_tokens"]:
            folder = ROOT / f"outputs/checkpoints/spatial_v1_{kind}_f{fold}"
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            model = ScratchMultiModalModel(config["modalities"], fusion=config["fusion"], visual_encoder=kind,
                                          visual_feedforward=config["visual_feedforward"], thermal_view=config["thermal_view"],
                                          imu_representation=config["imu_representation"]).cuda().eval()
            model.load_state_dict(checkpoint["model"], strict=True)
            indices = json.loads((folder / "split.json").read_text())["valid"]
            dataset = ClipDataset(indices, config["modalities"], frames=16, image_size=160, canonical_pose=True,
                                  thermal_view="motion_crop", imu_representation="world")
            loader = DataLoader(dataset, batch_size=24, num_workers=4, pin_memory=True)
            cases = ["baseline", "drop_skeleton", "drop_imu", "drop_thermal"]
            predictions = {case: [] for case in cases}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for batch in loader:
                    batch = prepare_batch(batch, torch.device("cuda"))
                    for case in cases:
                        altered = dict(batch)
                        if case.startswith("drop_"):
                            altered[case.removeprefix("drop_") + "_mask"] = torch.zeros_like(batch["skeleton_mask"])
                        output, _ = model(altered)
                        predictions[case].extend(output.argmax(1).cpu().tolist())
            labels = np.asarray([records[index]["label"] for index in indices])
            accuracy = {case: float((np.asarray(predictions[case]) == labels).mean()) for case in cases}
            with np.load(folder / "validation.npz") as saved:
                np.testing.assert_array_equal(saved["indices"], indices)
                replay_changes = int((np.asarray(predictions["baseline"]) != saved["probabilities"].argmax(1)).sum())
            assert replay_changes <= max(1, round(len(indices) * 0.01))
            weights, sizes = [], [1, 16, 16, 64 if kind == "cnn_pooled" else 1600]
            attention = model.transformer.layers[-1].multihead_attn
            original_forward = attention.forward

            def traced_forward(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                output, probabilities = original_forward(*args, **kwargs)
                masses = [part.float().sum(-1).mean((1, 2)).detach().cpu().numpy()
                          for part in probabilities.split(sizes, dim=-1)]
                weights.append(np.stack(masses, axis=1))
                return output, probabilities

            attention.forward = traced_forward
            traced_predictions = []
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for batch in loader:
                        batch = prepare_batch(batch, torch.device("cuda"))
                        output, _ = model(batch)
                        traced_predictions.extend(output.argmax(1).cpu().tolist())
            finally:
                attention.forward = original_forward
            trace = np.concatenate(weights)
            np.testing.assert_allclose(trace.sum(1), 1, atol=0.005)
            result = {"accuracy": accuracy, "baseline_replay_changed_labels": replay_changes,
                      "attention_trace_changed_labels": int((np.asarray(traced_predictions) != predictions["baseline"]).sum()),
                      "last_layer_mean_attention_mass": dict(zip(["null", "skeleton", "imu", "thermal"], trace.mean(0).tolist())),
                      "memory_tokens": dict(zip(["null", "skeleton", "imu", "thermal"], sizes)),
                      "all_modalities_uniform_token_prior": dict(zip(["null", "skeleton", "imu", "thermal"], (np.asarray(sizes) / sum(sizes)).tolist()))}
            report["experiments"][f"{kind}_f{fold}"] = result
            print(kind, fold, result, flush=True)
            del model, checkpoint, attention, original_forward
            torch.cuda.empty_cache()
    report["status"] = "complete"
    (ROOT / "outputs/study_spatial/fusion_diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
