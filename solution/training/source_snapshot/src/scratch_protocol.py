import numpy as np

from src.data import ROOT


def modality_presence(modalities):
    sensors = np.load(ROOT / "data/processed/sensor_mask.npy")
    video = np.load(ROOT / "data/processed/video_mask.npy")
    columns = {"skeleton": sensors[:, 0], "imu": sensors[:, 1], "imu_world": sensors[:, 1],
               "imu_relative": sensors[:, 1], "imu_official": sensors[:, 1],
               "imu_magnetic": sensors[:, 1], "radar": sensors[:, 2],
               "scene_depth": video[:, 0], "scene_ir": video[:, 1], "thermal": video[:, 2]}
    return np.stack([columns[name] > 0 for name in modalities], axis=1)


def classification_metrics(labels, prediction):
    confusion = np.zeros((40, 40), dtype=np.int64)
    np.add.at(confusion, (labels, prediction), 1)
    support = confusion.sum(1)
    recalls = np.diag(confusion) / support.clip(1)
    f1 = 2 * np.diag(confusion) / (support + confusion.sum(0)).clip(1)
    return {"accuracy": float((labels == prediction).mean()) if len(labels) else None,
            "macro_f1": float(f1[(support + confusion.sum(0)) > 0].mean()) if len(labels) else None,
            "macro_f1_all40": float(f1.mean()) if len(labels) else None,
            "balanced_accuracy": float(recalls[support > 0].mean()) if len(labels) else None,
            "class_support": support.tolist(), "class_recall": recalls.tolist(), "confusion_matrix": confusion.tolist()}


def ablation_metrics(predictions, records, modalities):
    indices = predictions["indices"]
    labels = predictions["labels"]
    prediction = predictions["probabilities"].argmax(1)
    presence = modality_presence(modalities)[indices]
    available = presence.any(1)
    common = modality_presence(["scene_ir", "thermal", "scene_depth", "skeleton", "imu", "radar"])[indices].all(1)
    users = np.asarray([records[index]["user"] for index in indices])
    return {"all_validation": classification_metrics(labels, prediction),
            "available_only": classification_metrics(labels[available], prediction[available]),
            "six_modality_common_subset": classification_metrics(labels[common], prediction[common]),
            "available_count": int(available.sum()), "validation_count": len(indices),
            "common_count": int(common.sum()), "coverage": float(available.mean()),
            "per_modality_coverage": dict(zip(modalities, presence.mean(0).tolist())),
            "per_user": {str(user): classification_metrics(labels[users == user], prediction[users == user])
                         for user in np.unique(users)}}
