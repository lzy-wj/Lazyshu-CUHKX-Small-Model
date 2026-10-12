#!/usr/bin/env python3
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from scripts.evaluate_repeat_decoder import rounds


class RoundResidualTransformer(nn.Module):
    def __init__(self, input_dimension=120, dimension=96, maximum_length=16):
        super().__init__()
        self.input_projection = nn.Linear(input_dimension, dimension)
        self.position_embedding = nn.Parameter(torch.zeros(1, maximum_length, dimension))
        self.domain_embedding = nn.Embedding(4, dimension)
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=4,
            dim_feedforward=dimension * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.normalization = nn.LayerNorm(dimension)
        self.residual = nn.Linear(dimension, 40)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features, anchor_log_probabilities, padding_mask, domains):
        tokens = self.input_projection(features)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        tokens = tokens + self.domain_embedding(domains).unsqueeze(1)
        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        correction = self.residual(self.normalization(tokens))
        return anchor_log_probabilities + correction


def normalized_log_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 1e-7, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.log(probabilities)


def make_batch(anchor, adapted, base, indices, domains, records, targets=None):
    indices = np.asarray(indices, dtype=np.int64)
    positions = {int(index): position for position, index in enumerate(indices)}
    round_values = rounds(indices, records)
    covered = {index for item in round_values for index in item["rows"]}
    round_values.extend(
        {"rows": [int(index)], "start": float("inf"), "end": float("inf")}
        for index in indices
        if int(index) not in covered
    )
    maximum_length = max(len(item["rows"]) for item in round_values)
    if maximum_length > 16:
        raise ValueError(f"Round length {maximum_length} exceeds the positional table")
    feature_values = np.concatenate(
        [
            normalized_log_probabilities(anchor),
            normalized_log_probabilities(adapted),
            normalized_log_probabilities(base),
        ],
        axis=1,
    )
    anchor_logs = normalized_log_probabilities(anchor)
    features = np.zeros((len(round_values), maximum_length, 120), dtype=np.float32)
    anchors = np.zeros((len(round_values), maximum_length, 40), dtype=np.float32)
    padding = np.ones((len(round_values), maximum_length), dtype=bool)
    domain_values = np.zeros(len(round_values), dtype=np.int64)
    target_values = np.full((len(round_values), maximum_length), -100, dtype=np.int64)
    output_positions = []
    for sequence_position, item in enumerate(round_values):
        row_positions = np.asarray([positions[int(index)] for index in item["rows"]])
        length = len(row_positions)
        if len(set(np.asarray(domains)[row_positions].tolist())) != 1:
            raise ValueError("A round crosses domain boundaries")
        features[sequence_position, :length] = feature_values[row_positions]
        anchors[sequence_position, :length] = anchor_logs[row_positions]
        padding[sequence_position, :length] = False
        domain_values[sequence_position] = int(np.asarray(domains)[row_positions[0]])
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
        "rounds": len(round_values),
    }


def fit(anchor, adapted, base, pseudo_labels, indices, domains, records, retention_weight=0.25,
        steps=600, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    batch = make_batch(anchor, adapted, base, indices, domains, records, pseudo_labels)
    model = RoundResidualTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    retained = batch["anchors"].exp()
    valid = ~batch["padding"]
    history = []
    exact_steps = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["features"], batch["anchors"], batch["padding"], batch["domains"])
        classification = functional.cross_entropy(
            logits.reshape(-1, 40), batch["targets"].reshape(-1), ignore_index=-100
        )
        retention = functional.kl_div(
            functional.log_softmax(logits[valid], dim=1),
            retained[valid],
            reduction="batchmean",
        )
        loss = classification + retention_weight * retention
        loss.backward()
        optimizer.step()
        correct = int((logits.detach().argmax(2)[valid] == batch["targets"][valid]).sum())
        exact_steps = exact_steps + 1 if correct == batch["rows"] else 0
        if step in (0, steps - 1) or (step + 1) % 100 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "classification": float(classification.detach()),
                    "retention": float(retention.detach()),
                    "teacher_imitation_correct": correct,
                }
            )
        if exact_steps >= 25:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "classification": float(classification.detach()),
                    "retention": float(retention.detach()),
                    "teacher_imitation_correct": correct,
                    "early_stop": True,
                }
            )
            break
    return model.eval(), batch, history


@torch.no_grad()
def predict(model, anchor, adapted, base, indices, domains, records):
    batch = make_batch(anchor, adapted, base, indices, domains, records)
    logits = model(batch["features"], batch["anchors"], batch["padding"], batch["domains"])
    probabilities = logits.softmax(dim=2).numpy()
    output = np.empty((batch["rows"], 40), dtype=np.float32)
    for sequence_position, row_positions in enumerate(batch["output_positions"]):
        output[row_positions] = probabilities[sequence_position, : len(row_positions)]
    return output, {"rounds": batch["rounds"], "rows": batch["rows"]}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bundle(model, path, teacher_path, retention_weight=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().half() for name, value in model.state_dict().items()}
    torch.save(
        {
            "format": "round_residual_transformer_v1",
            "state_dict": state,
            "retention_weight": retention_weight,
            "input": "round sequences of normalized V52, V24 adapted, and V24 base log probabilities",
            "teacher_sha256": file_sha256(teacher_path),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        path,
    )


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("format") != "round_residual_transformer_v1":
        raise ValueError("Unexpected round residual bundle format")
    model = RoundResidualTransformer()
    model.load_state_dict(
        {name: value.float() for name, value in bundle["state_dict"].items()},
        strict=True,
    )
    return model.eval(), bundle
