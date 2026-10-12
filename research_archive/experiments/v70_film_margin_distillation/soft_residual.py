#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from scripts.evaluate_repeat_decoder import rounds


class SoftRoundResidualTransformer(nn.Module):
    def __init__(
        self,
        expert_count=4,
        presence_dimension=6,
        dimension=128,
        maximum_length=16,
        layers=2,
    ):
        super().__init__()
        input_dimension = expert_count * 40 + presence_dimension
        self.input_projection = nn.Linear(input_dimension, dimension)
        self.position_embedding = nn.Parameter(torch.zeros(1, maximum_length, dimension))
        self.domain_embedding = nn.Embedding(4, dimension)
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=8,
            dim_feedforward=dimension * 3,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(dimension)
        self.residual = nn.Linear(dimension, 40)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features, anchor_logs, padding_mask, domains):
        values = self.input_projection(features)
        values = values + self.position_embedding[:, : values.shape[1]]
        values = values + self.domain_embedding(domains).unsqueeze(1)
        values = self.encoder(values, src_key_padding_mask=padding_mask)
        return anchor_logs + self.residual(self.normalization(values))


def normalized_logs(probabilities):
    values = np.asarray(probabilities, dtype=np.float32)
    values = np.clip(values, 1e-7, None)
    values /= values.sum(axis=1, keepdims=True)
    return np.log(values)


def make_batch(experts, indices, domains, presence, records, targets=None):
    indices = np.asarray(indices, dtype=np.int64)
    domains = np.asarray(domains, dtype=np.int64)
    presence = np.asarray(presence, dtype=np.float32)
    positions = {int(index): position for position, index in enumerate(indices)}
    sequences = []
    for user in sorted({records[int(index)]["user"] for index in indices}):
        user_indices = [int(index) for index in indices if records[int(index)]["user"] == user]
        user_rounds = rounds(user_indices, records)
        covered = {index for item in user_rounds for index in item["rows"]}
        user_rounds.extend(
            {"rows": [index], "start": float("inf"), "end": float("inf")}
            for index in user_indices
            if index not in covered
        )
        sequences.extend(user_rounds)
    maximum_length = max(len(item["rows"]) for item in sequences)
    if maximum_length > 16:
        raise ValueError(f"Round length {maximum_length} exceeds positional capacity")
    feature_values = np.concatenate(
        [normalized_logs(probabilities) for probabilities in experts] + [2 * presence - 1],
        axis=1,
    )
    anchor_values = normalized_logs(experts[0])
    features = np.zeros(
        (len(sequences), maximum_length, feature_values.shape[1]), dtype=np.float32
    )
    anchors = np.zeros((len(sequences), maximum_length, 40), dtype=np.float32)
    padding = np.ones((len(sequences), maximum_length), dtype=bool)
    domain_values = np.zeros(len(sequences), dtype=np.int64)
    target_values = np.zeros((len(sequences), maximum_length, 40), dtype=np.float32)
    output_positions = []
    for sequence_position, item in enumerate(sequences):
        row_positions = np.asarray([positions[int(index)] for index in item["rows"]])
        length = len(row_positions)
        if len(set(domains[row_positions].tolist())) != 1:
            raise ValueError("A sequence crosses domain boundaries")
        features[sequence_position, :length] = feature_values[row_positions]
        anchors[sequence_position, :length] = anchor_values[row_positions]
        padding[sequence_position, :length] = False
        domain_values[sequence_position] = int(domains[row_positions[0]])
        if targets is not None:
            target_values[sequence_position, :length] = np.asarray(targets)[row_positions]
        output_positions.append(row_positions)
    return {
        "features": torch.from_numpy(features),
        "anchors": torch.from_numpy(anchors),
        "padding": torch.from_numpy(padding),
        "domains": torch.from_numpy(domain_values),
        "targets": torch.from_numpy(target_values),
        "output_positions": output_positions,
        "rows": len(indices),
        "rounds": len(sequences),
    }


def fit(experts, teacher, indices, domains, presence, records, steps=1000, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    batch = make_batch(experts, indices, domains, presence, records, teacher)
    model = SoftRoundResidualTransformer(expert_count=len(experts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    valid = ~batch["padding"]
    target = batch["targets"][valid]
    anchor = batch["anchors"].exp()[valid]
    history = []
    stable_steps = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["features"], batch["anchors"], batch["padding"], batch["domains"])
        student_logs = functional.log_softmax(logits[valid], dim=1)
        divergence = functional.kl_div(student_logs, target, reduction="batchmean")
        centered_student = student_logs - student_logs.mean(dim=1, keepdim=True)
        target_logs = target.clamp_min(1e-7).log()
        centered_target = target_logs - target_logs.mean(dim=1, keepdim=True)
        logit_loss = functional.mse_loss(centered_student, centered_target)
        retention = functional.kl_div(student_logs, anchor, reduction="batchmean")
        loss = divergence + 0.05 * logit_loss + 0.01 * retention
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        with torch.no_grad():
            mean_absolute_error = float((student_logs.exp() - target).abs().mean())
        stable_steps = stable_steps + 1 if mean_absolute_error < 2e-4 else 0
        if step in (0, steps - 1) or (step + 1) % 100 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "kl": float(divergence.detach()),
                    "logit_loss": float(logit_loss.detach()),
                    "mean_absolute_error": mean_absolute_error,
                }
            )
        if stable_steps >= 30:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "kl": float(divergence.detach()),
                    "logit_loss": float(logit_loss.detach()),
                    "mean_absolute_error": mean_absolute_error,
                    "early_stop": True,
                }
            )
            break
    return model.eval(), history


@torch.no_grad()
def predict(model, experts, indices, domains, presence, records):
    batch = make_batch(experts, indices, domains, presence, records)
    logits = model(batch["features"], batch["anchors"], batch["padding"], batch["domains"])
    probabilities = logits.softmax(dim=2).numpy()
    output = np.empty((batch["rows"], 40), dtype=np.float32)
    for sequence_position, row_positions in enumerate(batch["output_positions"]):
        output[row_positions] = probabilities[sequence_position, : len(row_positions)]
    return output, {"rows": batch["rows"], "rounds": batch["rounds"]}


def save_bundle(model, path, teacher_description):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().half() for name, value in model.state_dict().items()}
    torch.save(
        {
            "format": "film_margin_round_transformer_v1",
            "state_dict": state,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "teacher": teacher_description,
            "inputs": ["V57", "V52", "V24 adapted", "V24 base"],
        },
        path,
    )


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("format") != "film_margin_round_transformer_v1":
        raise ValueError("Unexpected FiLM margin Transformer format")
    model = SoftRoundResidualTransformer(expert_count=len(bundle["inputs"]))
    model.load_state_dict(
        {name: value.float() for name, value in bundle["state_dict"].items()},
        strict=True,
    )
    return model.eval(), bundle
