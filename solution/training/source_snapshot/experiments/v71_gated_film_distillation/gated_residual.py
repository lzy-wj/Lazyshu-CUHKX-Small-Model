#!/usr/bin/env python3
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from experiments.v70_film_margin_distillation.soft_residual import (
    SoftRoundResidualTransformer,
    make_batch,
    predict,
)


class GatedFilmResidualTransformer(SoftRoundResidualTransformer):
    def __init__(
        self,
        expert_count=4,
        presence_dimension=6,
        dimension=256,
        maximum_length=16,
        layers=3,
        active_domain=2,
    ):
        super().__init__(
            expert_count=expert_count,
            presence_dimension=presence_dimension,
            dimension=dimension,
            maximum_length=maximum_length,
            layers=layers,
        )
        self.active_domain = active_domain

    def forward(self, features, anchor_logs, padding_mask, domains):
        logits = super().forward(features, anchor_logs, padding_mask, domains)
        gate = (domains == self.active_domain).to(logits.dtype).view(-1, 1, 1)
        return anchor_logs + gate * (logits - anchor_logs)


def fit_gated(
    experts,
    teacher,
    indices,
    domains,
    presence,
    records,
    steps=5000,
    seed=42,
    dimension=256,
    layers=3,
    active_domain=2,
):
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    batch = make_batch(experts, indices, domains, presence, records, teacher)
    model = GatedFilmResidualTransformer(
        expert_count=len(experts),
        dimension=dimension,
        layers=layers,
        active_domain=active_domain,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    sequence_active = batch["domains"] == active_domain
    active = (~batch["padding"]) & sequence_active.unsqueeze(1)
    target = batch["targets"][active]
    target_logs = target.clamp_min(1e-7).log()
    centered_target = target_logs - target_logs.mean(dim=1, keepdim=True)
    history = []
    stable_steps = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["features"], batch["anchors"], batch["padding"], batch["domains"]
        )
        student_logs = functional.log_softmax(logits[active], dim=1)
        divergence = functional.kl_div(student_logs, target, reduction="batchmean")
        centered_student = student_logs - student_logs.mean(dim=1, keepdim=True)
        logit_loss = functional.mse_loss(centered_student, centered_target)
        loss = divergence + 0.2 * logit_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        with torch.no_grad():
            mean_absolute_error = float((student_logs.exp() - target).abs().mean())
            maximum_absolute_error = float((student_logs.exp() - target).abs().max())
        stable_steps = stable_steps + 1 if mean_absolute_error < 1e-5 else 0
        if step in (0, steps - 1) or (step + 1) % 100 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "kl": float(divergence.detach()),
                    "logit_loss": float(logit_loss.detach()),
                    "mean_absolute_error": mean_absolute_error,
                    "maximum_absolute_error": maximum_absolute_error,
                }
            )
        if stable_steps >= 50:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "kl": float(divergence.detach()),
                    "logit_loss": float(logit_loss.detach()),
                    "mean_absolute_error": mean_absolute_error,
                    "maximum_absolute_error": maximum_absolute_error,
                    "early_stop": True,
                }
            )
            break
    return model.eval(), history


def save_bundle(model, path, teacher_description):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().half() for name, value in model.state_dict().items()}
    torch.save(
        {
            "format": "gated_film_margin_round_transformer_v1",
            "state_dict": state,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "teacher": teacher_description,
            "inputs": ["V57", "V52", "V24 adapted", "V24 base"],
            "dimension": 256,
            "layers": 3,
            "active_domain": model.active_domain,
        },
        path,
    )


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("format") != "gated_film_margin_round_transformer_v1":
        raise ValueError("Unexpected gated FiLM Transformer format")
    model = GatedFilmResidualTransformer(
        expert_count=len(bundle["inputs"]),
        dimension=bundle["dimension"],
        layers=bundle["layers"],
        active_domain=bundle["active_domain"],
    )
    model.load_state_dict(
        {name: value.float() for name, value in bundle["state_dict"].items()},
        strict=True,
    )
    return model.eval(), bundle


__all__ = ["fit_gated", "load_bundle", "predict", "save_bundle"]
