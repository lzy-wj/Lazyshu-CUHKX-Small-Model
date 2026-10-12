#!/usr/bin/env python3
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional


class LogitResidualConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Conv1d(40, 40, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, log_probabilities):
        correction = self.residual(log_probabilities.unsqueeze(-1)).squeeze(-1)
        return log_probabilities + correction


def normalized_log_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 1e-7, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return torch.from_numpy(np.log(probabilities))


def fit(probabilities, pseudo_labels, retention_weight, steps=300, seed=42):
    torch.manual_seed(seed)
    inputs = normalized_log_probabilities(probabilities)
    targets = torch.from_numpy(np.asarray(pseudo_labels, dtype=np.int64))
    retained = torch.from_numpy(np.asarray(probabilities, dtype=np.float32))
    model = LogitResidualConv()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.01)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        classification = functional.cross_entropy(logits, targets)
        retention = functional.kl_div(
            functional.log_softmax(logits, dim=1),
            retained,
            reduction="batchmean",
        )
        loss = classification + retention_weight * retention
        loss.backward()
        optimizer.step()
        if step in (0, steps - 1):
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "classification": float(classification.detach()),
                    "retention": float(retention.detach()),
                }
            )
    return model.eval(), history


@torch.no_grad()
def predict(model, probabilities):
    logits = model(normalized_log_probabilities(probabilities))
    return logits.softmax(dim=1).numpy()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(model, path, retention_weight, teacher_path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().half() for name, value in model.state_dict().items()}
    torch.save(
        {
            "format": "logit_residual_conv1d_v1",
            "state_dict": state,
            "retention_weight": retention_weight,
            "input": "log of normalized V24 TTA probabilities",
            "teacher_sha256": file_sha256(teacher_path),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        path,
    )


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "logit_residual_conv1d_v1":
        raise ValueError("Unsupported logit-residual checkpoint")
    model = LogitResidualConv()
    model.load_state_dict(
        {name: value.float() for name, value in payload["state_dict"].items()},
        strict=True,
    )
    return model.eval(), payload

