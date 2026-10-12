#!/usr/bin/env python3
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional


class DomainLogitResidualConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Conv1d(80, 40, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, adapted_log_probabilities, base_log_probabilities):
        features = torch.cat([adapted_log_probabilities, base_log_probabilities], dim=1)
        correction = self.residual(features.unsqueeze(-1)).squeeze(-1)
        return adapted_log_probabilities + correction


def normalized_log_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 1e-7, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return torch.from_numpy(np.log(probabilities))


def fit_one(adapted, base, pseudo_labels, retention_weight=0.25, steps=300, seed=42):
    torch.manual_seed(seed)
    adapted_inputs = normalized_log_probabilities(adapted)
    base_inputs = normalized_log_probabilities(base)
    targets = torch.from_numpy(np.asarray(pseudo_labels, dtype=np.int64))
    retained = torch.from_numpy(np.asarray(adapted, dtype=np.float32))
    model = DomainLogitResidualConv()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.01)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(adapted_inputs, base_inputs)
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
def predict_one(model, adapted, base):
    logits = model(
        normalized_log_probabilities(adapted),
        normalized_log_probabilities(base),
    )
    return logits.softmax(dim=1).numpy()


def fit_domains(adapted, base, pseudo_labels, groups, retention_weight=0.25):
    groups = np.asarray(groups)
    probabilities = np.empty_like(adapted)
    models = {}
    reports = {}
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        model, history = fit_one(
            adapted[selected],
            base[selected],
            pseudo_labels[selected],
            retention_weight=retention_weight,
        )
        probabilities[selected] = predict_one(model, adapted[selected], base[selected])
        models[group] = model
        reports[group] = {
            "rows": int(selected.sum()),
            "teacher_imitation_correct": int(
                (probabilities[selected].argmax(1) == pseudo_labels[selected]).sum()
            ),
            "history": history,
        }
    return probabilities, models, reports


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bundle(models, path, teacher_path, retention_weight=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    states = {
        group: {name: value.detach().half() for name, value in model.state_dict().items()}
        for group, model in models.items()
    }
    torch.save(
        {
            "format": "domain_logit_residual_conv1d_v1",
            "state_dicts": states,
            "retention_weight": retention_weight,
            "input": "concatenated logs of normalized V24 adapted and base TTA probabilities",
            "teacher_sha256": file_sha256(teacher_path),
            "parameters": sum(parameter.numel() for model in models.values() for parameter in model.parameters()),
        },
        path,
    )

