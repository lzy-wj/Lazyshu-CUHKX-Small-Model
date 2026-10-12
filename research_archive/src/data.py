import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
FOLDS = [[3, 8, 18, 23], [5, 7, 17, 22]]


def get_records():
    return json.loads((ROOT / "data/metadata/samples.json").read_text())


def get_indices(records, fold):
    train = [record["index"] for record in records if record["split"] == "train"
             and (fold < 0 or record["user"] not in FOLDS[fold])]
    valid = [record["index"] for record in records if record["split"] == "train"
             and fold >= 0 and record["user"] in FOLDS[fold]]
    if fold >= 0:
        if {records[index]["user"] for index in train} & {records[index]["user"] for index in valid}:
            raise RuntimeError("Subject leakage")
        if {records[index]["label"] for index in train} != set(range(40)):
            raise RuntimeError("Training split is missing classes")
    return train, valid


class ClipDataset(Dataset):
    def __init__(self, indices, modalities, training=False, frames=16, canonical_pose=False, image_size=96,
                 thermal_view="full", imu_representation="raw", temporal_crop=1.0):
        self.indices = indices
        self.modalities = modalities
        self.training = training
        self.frames = frames
        self.canonical_pose = canonical_pose
        self.image_size = image_size
        if not 0 < temporal_crop <= 1:
            raise ValueError("temporal_crop must be in (0, 1]")
        self.temporal_crop = temporal_crop
        if thermal_view not in ["full", "motion_crop"] or imu_representation not in ["raw", "world"]:
            raise ValueError("Unknown input representation")
        if thermal_view == "motion_crop" and image_size != 160:
            raise ValueError("Motion crop cache is prepared at 160 pixels")
        self.thermal_view = thermal_view
        self.records = get_records()
        self.sensors = {name: np.load(ROOT / f"data/processed/{name}.npy", mmap_mode="r")
                        for name in modalities if name in ["skeleton", "imu", "imu_world", "imu_timed", "imu_relative",
                                                           "imu_official", "imu_magnetic", "radar", "geometry"]}
        if "imu" in modalities and imu_representation == "world":
            self.sensors["imu"] = np.load(ROOT / "data/processed/imu_world.npy", mmap_mode="r")
        self.extra_masks = {name: np.load(ROOT / f"data/processed/{name}_mask.npy")
                            for name in ["imu_timed", "imu_relative"] if name in modalities}
        self.features = {name: np.load(ROOT / f"data/processed/{name}.npy", mmap_mode="r")
                         for name in modalities if name.endswith("_features")}
        self.feature_mask = np.load(ROOT / "data/processed/feature_mask.npy") if self.features else None
        self.sensor_mask = np.load(ROOT / "data/processed/sensor_mask.npy")
        self.video = any(name in modalities for name in ["depth_ir", "thermal", "scene_ir", "scene_depth"])
        self.video_mask = np.load(ROOT / "data/processed/video_mask.npy") if self.video else None
        self.person = any(name in ["person_ir", "person_thermal", "person_depth"] for name in modalities)
        self.person_mask = np.load(ROOT / "data/processed/person_mask.npy") if self.person else None
        self.detail = any(name in ["person_ir_detail", "person_thermal_detail"] for name in modalities)
        self.detail_mask = np.load(ROOT / "data/processed/person_detail_mask.npy") if self.detail else None
        self.yolo = any(name in ["yolo_rgb", "yolo_ir"] for name in modalities)
        self.yolo_frames = None
        if self.yolo:
            self.yolo_cache_dir = ROOT / "data/processed/r2plus1d_yolo_32_128"
            manifest = json.loads((self.yolo_cache_dir / "manifest.json").read_text())
            self.yolo_shape = tuple(manifest["shape"])
            if self.yolo_shape != (len(self.records), 32, 4, 128, 128) or not manifest.get("complete"):
                raise ValueError("YOLO RGB/IR cache is incomplete or has an unexpected shape")

    def __len__(self):
        return len(self.indices)

    def sample_positions(self, length, temporal_jitter=None):
        margin = (1 - self.temporal_crop) / 2
        positions = np.linspace(margin, 1 - margin, self.frames)
        if temporal_jitter is not None:
            positions += temporal_jitter
        return np.clip(np.round(positions * (length - 1)), 0, length - 1).astype(int)

    def __getitem__(self, position):
        index = self.indices[position]
        result = {"label": self.records[index]["label"], "index": index}
        temporal_jitter = np.random.uniform(-0.04, 0.04, self.frames) if self.training else None
        for name, values in self.features.items():
            result[name] = np.array(values[index])
            result[name + "_mask"] = self.feature_mask[index, ["ir_features", "thermal_features", "depth_features"].index(name)]
        if self.person:
            video = np.load(ROOT / f"data/processed/person_16_{self.image_size}/{index:05d}.npy", mmap_mode="r")
            positions = self.sample_positions(len(video), temporal_jitter)
            for modality_index, name in enumerate(["person_ir", "person_thermal", "person_depth"]):
                if name in self.modalities:
                    result[name] = np.array(video[positions, modality_index * 3:(modality_index + 1) * 3])
                    result[name + "_mask"] = self.person_mask[index, modality_index]
        if self.detail:
            video = np.load(ROOT / f"data/processed/person_detail_16_{self.image_size}/{index:05d}.npy", mmap_mode="r")
            positions = self.sample_positions(len(video), temporal_jitter)
            for modality_index, name in enumerate(["person_ir_detail", "person_thermal_detail"]):
                if name in self.modalities:
                    result[name] = np.array(video[positions, modality_index * 3:(modality_index + 1) * 3])
                    result[name + "_mask"] = self.detail_mask[index, modality_index]
        if self.yolo:
            if self.yolo_frames is None:
                self.yolo_frames = np.memmap(
                    self.yolo_cache_dir / "frames.uint8.memmap",
                    dtype=np.uint8,
                    mode="r",
                    shape=self.yolo_shape,
                )
            positions = self.sample_positions(self.yolo_shape[1], temporal_jitter)
            video = np.asarray(self.yolo_frames[index, positions]).copy()
            if "yolo_rgb" in self.modalities:
                result["yolo_rgb"] = video[:, :3]
                result["yolo_rgb_mask"] = np.float32(video[:, :3].any())
            if "yolo_ir" in self.modalities:
                result["yolo_ir"] = np.repeat(video[:, 3:4], 3, axis=1)
                result["yolo_ir_mask"] = np.float32(video[:, 3:4].any())
        for name, values in self.sensors.items():
            result[name] = np.array(values[index])
            if name == "skeleton" and self.canonical_pose:
                pose = result[name][:, :51].reshape(64, 17, 3).copy()
                pose[..., :2] -= pose[:, :1, :2]
                lateral = (pose[:, 1, :2] - pose[:, 4, :2] + pose[:, 14, :2] - pose[:, 11, :2]).mean(0)
                angle = np.arctan2(lateral[1], lateral[0])
                horizontal = pose[..., 0] * np.cos(angle) + pose[..., 1] * np.sin(angle)
                vertical = -pose[..., 0] * np.sin(angle) + pose[..., 1] * np.cos(angle)
                pose[..., 0], pose[..., 1] = horizontal, vertical
                scale = np.clip(np.median(np.linalg.norm(pose[:, 8] - pose[:, 0], axis=-1)), 0.15, 1.2)
                pose = pose / scale * 0.5
                bones = pose - pose[:, [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]]
                velocity = np.diff(pose, axis=0, prepend=pose[:1]) * 10
                result[name] = np.concatenate([pose.reshape(64, -1), bones.reshape(64, -1),
                                               velocity.reshape(64, -1)], axis=1)
            mask_name = {"geometry": "skeleton", "imu_world": "imu", "imu_timed": "imu", "imu_relative": "imu",
                         "imu_official": "imu", "imu_magnetic": "imu"}.get(name, name)
            result[name + "_mask"] = (self.extra_masks[name][index] if name in self.extra_masks else
                                      self.sensor_mask[index, ["skeleton", "imu", "radar"].index(mask_name)])
        if self.video:
            video = np.load(ROOT / f"data/processed/video_24_{self.image_size}/{index:05d}.npy", mmap_mode="r")
            positions = self.sample_positions(len(video), temporal_jitter)
            selected = np.array(video[positions])
            if "depth_ir" in self.modalities:
                result["depth_ir"] = selected[:, :4]
                result["depth_ir_mask"] = self.video_mask[index, :2].max()
            if "scene_ir" in self.modalities:
                result["scene_ir"] = np.repeat(selected[:, 3:4], 3, axis=1)
                result["scene_ir_mask"] = self.video_mask[index, 1]
            if "scene_depth" in self.modalities:
                result["scene_depth"] = selected[:, :3]
                result["scene_depth_mask"] = self.video_mask[index, 0]
            if "thermal" in self.modalities:
                if self.thermal_view == "motion_crop":
                    cropped = np.load(ROOT / f"data/processed/thermal_motion_24_160/{index:05d}.npy", mmap_mode="r")
                    result["thermal"] = np.array(cropped[positions])
                else:
                    result["thermal"] = selected[:, 4:]
                result["thermal_mask"] = self.video_mask[index, 2]
        return result


def prepare_batch(batch, device, training=False, modality_dropout=0.1, strong_augment=False,
                  spatial_augment=False, synchronized_temporal_augment=False):
    result = {}
    temporal_window = None
    if training and strong_augment and synchronized_temporal_augment:
        temporal_window = (
            float(torch.empty(()).uniform_(0, 9 / 64)),
            float(torch.empty(()).uniform_(54 / 64, 1)),
        )
    for name, values in batch.items():
        values = values.to(device, non_blocking=True)
        if name in ["depth_ir", "thermal", "scene_ir", "scene_depth", "person_ir", "person_thermal", "person_depth", "person_ir_detail", "person_thermal_detail", "yolo_rgb", "yolo_ir"]:
            values = values.float().div_(255)
            if training:
                batch_size, steps, channels, height, width = values.shape
                brightness = 1 + torch.randn(batch_size, 1, 1, 1, 1, device=device) * 0.08
                values = (values * brightness).clamp_(0, 1)
                offset = torch.randint(-4, 5, (2,)).tolist()
                values = torch.roll(values, offset, dims=(-2, -1))
                if spatial_augment:
                    transform = torch.zeros(batch_size, 2, 3, device=device)
                    scale = torch.empty(batch_size, device=device).uniform_(0.65, 1)
                    flip = torch.where(torch.rand(batch_size, device=device) < 0.5, -1.0, 1.0)
                    transform[:, 0, 0] = scale * flip
                    transform[:, 1, 1] = scale
                    transform[:, :, 2] = (torch.rand(batch_size, 2, device=device) * 2 - 1) * (1 - scale[:, None])
                    grid = torch.nn.functional.affine_grid(transform, (batch_size, channels, height, width), align_corners=False)
                    values = torch.nn.functional.grid_sample(values.flatten(0, 1), grid.repeat_interleave(steps, 0),
                                                             align_corners=False, padding_mode="border").reshape_as(values)
                if temporal_window is not None:
                    start = round(temporal_window[0] * (steps - 1))
                    end = max(start + 2, round(temporal_window[1] * (steps - 1)) + 1)
                    values = torch.nn.functional.interpolate(
                        values[:, start:end].permute(0, 2, 1, 3, 4),
                        size=(steps, height, width),
                        mode="trilinear",
                        align_corners=False,
                    ).permute(0, 2, 1, 3, 4)
            values = (values - 0.5) * 2
        elif name.endswith("_features") and training:
            values = values + torch.randn_like(values) * 0.01
            values = torch.nn.functional.dropout(values, p=0.15, training=True)
        elif training and name in ["skeleton", "imu", "imu_world", "imu_timed", "imu_relative",
                                   "imu_official", "imu_magnetic", "radar", "geometry"]:
            values = values + torch.randn_like(values) * 0.008
            if strong_augment:
                count, steps, channels = values.shape
                if temporal_window is None:
                    start = int(torch.randint(0, 10, ()).item())
                    end = int(torch.randint(steps - 10, steps + 1, ()).item())
                else:
                    start = round(temporal_window[0] * (steps - 1))
                    end = max(start + 2, round(temporal_window[1] * (steps - 1)) + 1)
                values = torch.nn.functional.interpolate(values[:, start:end].transpose(1, 2),
                                                         size=steps, mode="linear", align_corners=True).transpose(1, 2)
                if name == "skeleton":
                    pose = values[:, :, :51].reshape(count, steps, 17, 3).clone()
                    angle = torch.randn(count, 1, 1, device=device) * 0.25
                    horizontal = pose[..., 0] * angle.cos() - pose[..., 1] * angle.sin()
                    vertical = pose[..., 0] * angle.sin() + pose[..., 1] * angle.cos()
                    pose[..., 0], pose[..., 1] = horizontal, vertical
                    pose = pose * (1 + torch.randn(count, 1, 1, 1, device=device) * 0.10)
                    bones = pose - pose[:, :, [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]]
                    velocity = torch.diff(pose, dim=1, prepend=pose[:, :1]) * 10
                    values = torch.cat([pose.flatten(2), bones.flatten(2), velocity.flatten(2)], dim=2)
                elif name == "imu":
                    keep = (torch.rand(count, 1, 5, 1, device=device) > 0.15).float()
                    features = values[:, :, :80].reshape(count, steps, 5, 16) * keep
                    values = torch.cat([features.flatten(2), values[:, :, 80:]], dim=2)
        result[name] = values
    if training and modality_dropout > 0:
        names = [name for name in result if name.endswith("_mask")]
        if len(names) > 1:
            original = torch.stack([result[name] for name in names], 1)
            dropped = original * (torch.rand_like(original) > modality_dropout)
            keep_original = dropped.sum(1) == 0
            dropped[keep_original] = original[keep_original]
            for position, name in enumerate(names):
                result[name] = dropped[:, position]
    return result
