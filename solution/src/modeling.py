import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.r2plus1d_models import R2Plus1DClassifier, dequantize_state_dict
from src.runtime_data import RuntimeDataset


def checkpoint_state(checkpoint):
    for key in ("model_int4", "model_int7", "model_int6", "model_int8"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def build_base_model(config):
    return R2Plus1DClassifier(
        mode=config["mode"],
        weights=None,
        dropout=float(config.get("dropout", 0.3)),
        input_channels=int(config.get("input_channels", 7)),
        mask_joint_visual=bool(config.get("mask_joint_visual", False)),
        use_radar=bool(config.get("use_radar", False)),
        skeleton_representation=config.get("skeleton_representation", "coordinates"),
        mixstyle_probability=float(config.get("mixstyle_probability", 0.0)),
        mixstyle_alpha=float(config.get("mixstyle_alpha", 0.1)),
        fusion_depth=int(config.get("fusion_depth", 2)),
        use_temporal_embedding=bool(config.get("use_temporal_embedding", False)),
        use_phase_temporal_embedding=bool(config.get("use_phase_temporal_embedding", False)),
        fusion_output=config.get("fusion_output", "residual"),
        visual_motion_channels=bool(config.get("visual_motion_channels", False)),
        visual_strategy=config.get("visual_strategy", "sequential"),
        per_visual_auxiliary=bool(config.get("per_visual_auxiliary", False)),
        environment_moe=float(config.get("environment_auxiliary_weight", 0.0)) > 0,
        visual_multiscale_tokens=bool(config.get("visual_multiscale_tokens", False)),
        normalize_modality_tokens=bool(config.get("normalize_modality_tokens", False)),
        skeleton_encoder=config.get("skeleton_encoder", "tcn"),
        radar_representation=config.get("radar_representation", "statistics"),
        imu_inputs={
            "world": 85,
            "relative": 95,
            "combined": 180,
            "world_magnetic": 100,
            "world_official": 135,
        }[config.get("imu_representation", "world")],
        imu_statistics_residual=bool(config.get("imu_statistics_residual", False)),
        imu_statistics_logit_residual=bool(
            config.get("imu_statistics_logit_residual", False)
        ),
        imu_device_attention_residual=bool(
            config.get("imu_device_attention_residual", False)
        ),
        imu_statistics_inputs=50
        if config.get("imu_representation") == "world_official"
        else None,
    )


def apply_head_delta(state, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    output = {name: value.clone() for name, value in state.items()}
    for name, delta in payload["delta"].items():
        if name not in output or output[name].shape != delta.shape:
            raise ValueError(f"Head delta tensor mismatch: {name}")
        output[name] = output[name] + delta.to(output[name].dtype)
    return output


class SpatiotemporalResidualAdapter(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        groups = min(8, bottleneck)
        while bottleneck % groups:
            groups -= 1
        self.down = nn.Conv3d(channels, bottleneck, kernel_size=1, bias=False)
        self.normalization = nn.GroupNorm(groups, bottleneck)
        self.activation = nn.GELU()
        self.context = nn.Conv3d(
            bottleneck,
            bottleneck,
            kernel_size=3,
            padding=1,
            groups=bottleneck,
            bias=False,
        )
        self.context_normalization = nn.GroupNorm(groups, bottleneck)
        self.up = nn.Conv3d(bottleneck, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values):
        residual = self.activation(self.normalization(self.down(values)))
        residual = self.activation(self.context_normalization(self.context(residual)))
        return values + self.up(residual)


class ProtectedAdapterVideoEncoder(nn.Module):
    def __init__(self, source, bottleneck=128, output_adapters=True):
        super().__init__()
        self.stem = source.stem
        self.layer1 = source.layer1
        self.layer2 = source.layer2
        self.layer3 = source.layer3
        self.layer4 = source.layer4
        self.intermediate_adapters = nn.ModuleList(
            [SpatiotemporalResidualAdapter(256, bottleneck) for _ in range(3)]
        )
        self.output_adapters = nn.ModuleList(
            [SpatiotemporalResidualAdapter(512, bottleneck) for _ in range(3)]
            if output_adapters
            else []
        )

    def forward_joint(self, videos, return_multiscale=False):
        values = self.stem(torch.cat(videos, dim=0))
        values = self.layer2(self.layer1(values))
        intermediate = self.layer3(values)
        intermediate_streams = [
            adapter(stream)
            for adapter, stream in zip(
                self.intermediate_adapters, intermediate.chunk(3, dim=0)
            )
        ]
        output = self.layer4(torch.cat(intermediate_streams, dim=0))
        output_streams = list(output.chunk(3, dim=0))
        if self.output_adapters:
            output_streams = [
                adapter(stream)
                for adapter, stream in zip(self.output_adapters, output_streams)
            ]
        if return_multiscale:
            return list(zip(intermediate_streams, output_streams))
        return output_streams

    def forward(self, video, modality=0, return_multiscale=False):
        values = self.layer2(self.layer1(self.stem(video)))
        intermediate = self.intermediate_adapters[modality](self.layer3(values))
        output = self.layer4(intermediate)
        if self.output_adapters:
            output = self.output_adapters[modality](output)
        return (intermediate, output) if return_multiscale else output


class ProtectedVisualAdapterClassifier(nn.Module):
    def __init__(self, base, bottleneck=128, output_adapters=True):
        super().__init__()
        base.video = ProtectedAdapterVideoEncoder(
            base.video,
            bottleneck=bottleneck,
            output_adapters=output_adapters,
        )
        base.per_visual_auxiliary = True
        self.base = base

    def forward(
        self,
        video,
        skeleton=None,
        imu=None,
        radar=None,
        sensor_mask=None,
        visual_mask=None,
    ):
        return self.base(
            video,
            skeleton=skeleton,
            imu=imu,
            radar=radar,
            sensor_mask=sensor_mask,
            visual_mask=visual_mask,
        )


def make_loader(
    indices,
    records,
    sensors_path,
    video_path,
    image_size,
    batch_size,
    workers,
):
    dataset = RuntimeDataset(indices, records, sensors_path, video_path, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def model_inputs(batch, device):
    video = batch["video"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4)
    keyword = {
        "visual_mask": batch["visual_mask"].to(device, non_blocking=True),
        "skeleton": batch["skeleton"].to(device, non_blocking=True),
        "imu": batch["imu"].to(device, non_blocking=True),
        "sensor_mask": batch["sensor_mask"].to(device, non_blocking=True),
    }
    return video, keyword


@torch.no_grad()
def adapt_batch_norm(model, loader, device):
    model.eval()
    batch_norms = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
            batch_norms.append(layer)
    if not batch_norms:
        raise RuntimeError("AdaBN requested for a model without batch normalization")
    for batch in tqdm(loader, desc="AdaBN"):
        video, keyword = model_inputs(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(video, **keyword)
    model.eval()


@torch.no_grad()
def predict(model, loader, device, tta_hflip, crop_scale):
    output = {}
    for batch in tqdm(loader, desc="Inference"):
        video, keyword = model_inputs(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(video, **keyword)
            views = [logits]
            if tta_hflip:
                views.append(model(torch.flip(video, dims=(-1,)), **keyword)[0])
            if crop_scale < 1:
                side = round(video.shape[-1] * crop_scale)
                top = (video.shape[-2] - side) // 2
                left = (video.shape[-1] - side) // 2
                cropped = functional.interpolate(
                    video[..., top : top + side, left : left + side],
                    size=video.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                views.append(model(cropped, **keyword)[0])
                if tta_hflip:
                    views.append(model(torch.flip(cropped, dims=(-1,)), **keyword)[0])
            probabilities = torch.stack(views).mean(0).float().softmax(1).cpu().numpy()
        for index, values in zip(batch["index"].numpy(), probabilities):
            output[int(index)] = values
    return output


def infer_base_variant(
    checkpoint_path,
    delta_path,
    indices,
    records,
    sensors_path,
    video_path,
    image_size,
    batch_size,
    workers,
    device,
    adabn,
    tta_hflip=True,
    crop_scale=0.75,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint_state(checkpoint)
    if delta_path is not None:
        state = apply_head_delta(state, delta_path)
    model = build_base_model(checkpoint["config"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    loader = make_loader(
        indices,
        records,
        sensors_path,
        video_path,
        image_size,
        batch_size,
        workers,
    )
    if adabn:
        adapt_batch_norm(model, loader, device)
    values = predict(model, loader, device, tta_hflip, crop_scale)
    result = np.stack([values[int(index)] for index in indices])
    del loader, model
    torch.cuda.empty_cache()
    return result


def infer_protected_variant(
    checkpoint_path,
    adapter_path,
    payload_key,
    indices,
    records,
    sensors_path,
    video_path,
    batch_size,
    workers,
    device,
    tta_hflip,
    crop_scale,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    adapter_config = payload["config"]
    bottleneck = int(adapter_config.get("bottleneck", 128))
    output_adapters = adapter_config.get("location", "dual") == "dual"
    if payload_key == "state":
        output_adapters = True
    base = build_base_model(checkpoint["config"])
    base.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model = ProtectedVisualAdapterClassifier(
        base,
        bottleneck=bottleneck,
        output_adapters=output_adapters,
    )
    result = model.load_state_dict(payload[payload_key], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter tensors: {result.unexpected_keys}")
    model.to(device).eval()
    loader = make_loader(
        indices,
        records,
        sensors_path,
        video_path,
        160,
        batch_size,
        workers,
    )
    values = predict(model, loader, device, tta_hflip, crop_scale)
    output = np.stack([values[int(index)] for index in indices])
    del loader, model
    torch.cuda.empty_cache()
    return output
