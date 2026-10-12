import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import Dataset, get_worker_info

from src.data import ROOT, get_records


KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class R2Plus1DDataset(Dataset):
    def __init__(self, indices, cache_dir, training=False, include_sensors=False, include_radar=False,
                 image_size=128, frames=16, photometric_jitter=0.0, canonical_pose=False,
                 horizontal_flip_probability=0.5, synchronized_horizontal_flip=False,
                 skeleton_scale_randomization=False, photometric_gamma=0.0,
                 radar_representation="statistics", imu_representation="world"):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.cache_dir = Path(cache_dir)
        self.training = training
        self.include_sensors = include_sensors
        self.include_radar = include_radar
        self.radar_representation = radar_representation
        self.imu_representation = imu_representation
        self.image_size = image_size
        self.frames = frames
        self.photometric_jitter = photometric_jitter
        self.photometric_gamma = photometric_gamma
        self.canonical_pose = canonical_pose
        self.horizontal_flip_probability = horizontal_flip_probability
        self.synchronized_horizontal_flip = synchronized_horizontal_flip
        self.skeleton_scale_randomization = skeleton_scale_randomization
        self.skeleton_scale_rng = None
        if not 0 <= photometric_jitter <= 0.5:
            raise ValueError("Photometric jitter must be in [0, 0.5]")
        if not 0 <= photometric_gamma <= 1:
            raise ValueError("Photometric gamma range must be in [0, 1]")
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("Horizontal flip probability must be in [0, 1]")
        if radar_representation not in ("statistics", "points"):
            raise ValueError(f"Unknown radar representation: {radar_representation}")
        if imu_representation not in ("world", "relative", "combined", "world_magnetic", "world_official"):
            raise ValueError(f"Unknown IMU representation: {imu_representation}")
        if synchronized_horizontal_flip and imu_representation != "world":
            raise ValueError("Synchronized flipping is not implemented for relative IMU")
        self.records = get_records()
        manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.shape = tuple(manifest["shape"])
        if self.shape[0] != len(self.records) or self.shape[1] != frames or self.shape[3:] != (image_size, image_size):
            raise ValueError("R(2+1)D cache shape does not match requested dataset")
        self.channels = self.shape[2]
        expected = int(np.prod(self.shape))
        cache_path = self.cache_dir / "frames.uint8.memmap"
        if not cache_path.is_file() or cache_path.stat().st_size != expected:
            raise ValueError(f"Incomplete R(2+1)D cache: {cache_path}")
        if not manifest.get("complete") or manifest["shape"] != list(self.shape):
            raise ValueError("R(2+1)D cache manifest does not match requested shape")
        self.cache_path = str(cache_path)
        rgb_streams = len(manifest.get("rgb_modalities", ["Depth_Color"]))
        self.rgb_streams = rgb_streams
        expected_channels = 3 * rgb_streams + 1
        if self.channels != expected_channels:
            raise ValueError("Cache modalities and channel count disagree")
        mean = KINETICS_MEAN * rgb_streams + (sum(KINETICS_MEAN) / 3,)
        std = KINETICS_STD * rgb_streams + (sum(KINETICS_STD) / 3,)
        self.mean = torch.tensor(mean).view(1, self.channels, 1, 1)
        self.std = torch.tensor(std).view(1, self.channels, 1, 1)
        self.frames_memmap = None
        if include_sensors:
            self.skeleton = np.load(ROOT / "data/processed/skeleton.npy", mmap_mode="r")
            world_imu = np.load(ROOT / "data/processed/imu_world.npy", mmap_mode="r")
            relative_imu = (
                np.load(ROOT / "data/processed/imu_relative.npy", mmap_mode="r")
                if imu_representation in ("relative", "combined")
                else None
            )
            self.imu = (
                world_imu
                if imu_representation == "world"
                else relative_imu
                if imu_representation == "relative"
                else (world_imu, relative_imu)
                if imu_representation == "combined"
                else np.load(ROOT / f"data/processed/imu_{imu_representation}.npy", mmap_mode="r")
            )
            self.imu_mask = (
                np.load(ROOT / "data/processed/imu_relative_mask.npy", mmap_mode="r")
                if imu_representation == "relative"
                else None
            )
            radar_path = (
                ROOT / "data/processed/radar_points.npy"
                if radar_representation == "points"
                else ROOT / "data/processed/radar.npy"
            )
            self.radar = np.load(radar_path, mmap_mode="r") if include_radar else None
            self.sensor_mask = np.load(ROOT / "data/processed/sensor_mask.npy", mmap_mode="r")
            self.skeleton_scales = None
            if training and skeleton_scale_randomization:
                scales = []
                for index in self.indices:
                    pose = np.asarray(self.skeleton[int(index), :, :51]).reshape(64, 17, 3)
                    scales.append(np.median(np.linalg.norm(pose[:, 8] - pose[:, 0], axis=-1)))
                self.skeleton_scales = np.asarray(scales, dtype=np.float32).clip(0.15, 1.2)

    def __len__(self):
        return len(self.indices)

    @property
    def cache(self):
        if self.frames_memmap is None:
            self.frames_memmap = np.memmap(self.cache_path, dtype=np.uint8, mode="r", shape=self.shape)
        return self.frames_memmap

    def augment_video(self, video):
        if self.photometric_jitter > 0 or self.photometric_gamma > 0:
            stream_slices = [slice(stream * 3, (stream + 1) * 3) for stream in range(self.rgb_streams)]
            stream_slices.append(slice(self.rgb_streams * 3, self.channels))
            for channels in stream_slices:
                if self.photometric_gamma > 0:
                    gamma = math.exp(random.uniform(-self.photometric_gamma, self.photometric_gamma))
                    video[:, channels].pow_(gamma)
                if self.photometric_jitter > 0:
                    gain = 1 + random.uniform(-self.photometric_jitter, self.photometric_jitter)
                    offset = random.uniform(-self.photometric_jitter / 4, self.photometric_jitter / 4)
                    video[:, channels].mul_(gain).add_(offset).clamp_(0, 1)
        flipped = random.random() < self.horizontal_flip_probability
        if flipped:
            video = torch.flip(video, dims=(-1,))
        scale = random.uniform(0.75, 1.0)
        side = round(self.image_size * scale)
        top = random.randint(0, self.image_size - side)
        left = random.randint(0, self.image_size - side)
        return functional.interpolate(
            video[:, :, top:top + side, left:left + side],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ), flipped

    @staticmethod
    def mirror_skeleton(skeleton):
        joint_order = torch.tensor(
            [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13],
            device=skeleton.device,
        )
        values = skeleton.reshape(skeleton.shape[0], 3, 17, 3)[:, :, joint_order].clone()
        values[..., 0].neg_()
        return values.reshape_as(skeleton)

    @staticmethod
    def mirror_imu(imu):
        device_order = torch.tensor([1, 0, 2, 4, 3], device=imu.device)
        values = imu[:, :80].reshape(imu.shape[0], 5, 16)[:, device_order].clone()
        values[..., [0, 6]] = -values[..., [0, 6]]
        values[..., [4, 5, 10, 11]] = -values[..., [4, 5, 10, 11]]
        presence = imu[:, 80:85][:, device_order].clone()
        return torch.cat([values.reshape(imu.shape[0], 80), presence], dim=1)

    @staticmethod
    def transfer_skeleton_scale(skeleton, target_scale):
        pose = skeleton[:, :51].reshape(skeleton.shape[0], 17, 3)
        root = pose[:, :1]
        current_scale = torch.linalg.vector_norm(pose[:, 8] - pose[:, 0], dim=-1).median().clamp(0.15, 1.2)
        pose = root + (pose - root) * (target_scale / current_scale)
        parents = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        bones = pose - pose[:, parents]
        velocity = torch.diff(pose, dim=0, prepend=pose[:1]) * 10
        return torch.cat([pose.flatten(1), bones.flatten(1), velocity.flatten(1)], dim=1).clamp(-10, 10)

    def sample_skeleton_scale(self):
        if self.skeleton_scale_rng is None:
            worker = get_worker_info()
            seed = worker.seed if worker is not None else torch.initial_seed()
            self.skeleton_scale_rng = random.Random(seed ^ 0x5A17E5CA1E)
        return float(self.skeleton_scales[self.skeleton_scale_rng.randrange(len(self.skeleton_scales))])

    @staticmethod
    def canonicalize_skeleton(skeleton):
        pose = skeleton[:, :51].reshape(64, 17, 3).clone()
        pose[..., :2] -= pose[:, :1, :2]
        lateral = (pose[:, 1, :2] - pose[:, 4, :2] + pose[:, 14, :2] - pose[:, 11, :2]).mean(0)
        angle = torch.atan2(lateral[1], lateral[0])
        horizontal = pose[..., 0] * angle.cos() + pose[..., 1] * angle.sin()
        vertical = -pose[..., 0] * angle.sin() + pose[..., 1] * angle.cos()
        pose[..., 0] = horizontal
        pose[..., 1] = vertical
        scale = torch.linalg.vector_norm(pose[:, 8] - pose[:, 0], dim=-1).median().clamp(0.15, 1.2)
        pose = pose / scale * 0.5
        parents = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        bones = pose - pose[:, parents]
        velocity = torch.diff(pose, dim=0, prepend=pose[:1]) * 10
        return torch.cat([pose.flatten(1), bones.flatten(1), velocity.flatten(1)], dim=1)

    def __getitem__(self, position):
        index = int(self.indices[position])
        video = torch.from_numpy(np.asarray(self.cache[index]).copy()).float().div_(255)
        visual_mask = []
        for stream in range(self.rgb_streams):
            visual_mask.append(video[:, stream * 3:(stream + 1) * 3].any().float())
        visual_mask.append(video[:, self.rgb_streams * 3:].any().float())
        flipped = False
        if self.training:
            video, flipped = self.augment_video(video)
        item = {
            "video": video.sub_(self.mean).div_(self.std),
            "label": int(self.records[index]["label"]),
            "index": index,
            "visual_mask": torch.stack(visual_mask),
        }
        if self.include_sensors:
            skeleton = torch.from_numpy(np.asarray(self.skeleton[index]).copy())
            if self.canonical_pose:
                skeleton = self.canonicalize_skeleton(skeleton)
            if self.imu_representation == "combined":
                imu = torch.from_numpy(
                    np.concatenate(
                        [np.asarray(values[index]) for values in self.imu], axis=1
                    ).copy()
                )
            else:
                imu = torch.from_numpy(np.asarray(self.imu[index]).copy())
            if self.training and self.skeleton_scales is not None:
                skeleton = self.transfer_skeleton_scale(skeleton, self.sample_skeleton_scale())
            if flipped and self.synchronized_horizontal_flip:
                skeleton = self.mirror_skeleton(skeleton)
                imu = self.mirror_imu(imu)
            radar = torch.from_numpy(np.asarray(self.radar[index]).copy()) if self.include_radar else None
            if self.training:
                skeleton = skeleton + torch.randn_like(skeleton) * 0.006
                imu = imu + torch.randn_like(imu) * 0.006
                if radar is not None:
                    radar = radar + torch.randn_like(radar) * 0.006
            sensor_mask = np.asarray(
                self.sensor_mask[index, :3 if self.include_radar else 2]
            ).copy()
            if self.imu_mask is not None:
                sensor_mask[1] = self.imu_mask[index]
            item.update(
                {
                    "skeleton": skeleton,
                    "imu": imu,
                    "sensor_mask": torch.from_numpy(sensor_mask),
                }
            )
            if radar is not None:
                item["radar"] = radar
        return item
