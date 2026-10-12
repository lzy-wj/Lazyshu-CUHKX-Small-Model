#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from scripts.evaluate_repeat_decoder import rounds


class ExpertRoundTransformer(nn.Module):
    def __init__(self, expert_count=6, presence_dimension=6, dimension=64, maximum_length=16):
        super().__init__()
        input_dimension = expert_count * 40 + presence_dimension
        self.input_projection = nn.Linear(input_dimension, dimension)
        self.position_embedding = nn.Parameter(torch.zeros(1, maximum_length, dimension))
        self.domain_embedding = nn.Embedding(4, dimension)
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=4,
            dim_feedforward=dimension * 2,
            dropout=0.1,
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

    def forward(self, features, anchor_logs, padding_mask, domains):
        tokens = self.input_projection(features)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        tokens = tokens + self.domain_embedding(domains).unsqueeze(1)
        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        correction = self.residual(self.normalization(tokens))
        return anchor_logs + correction


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
    users = sorted({records[int(index)]["user"] for index in indices})
    for user in users:
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
    target_values = np.full((len(sequences), maximum_length), -100, dtype=np.int64)
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


def fit(
    experts,
    labels,
    indices,
    domains,
    presence,
    records,
    retention_weight,
    steps=300,
    seed=42,
):
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 12))
    batch = make_batch(experts, indices, domains, presence, records, labels)
    model = ExpertRoundTransformer(expert_count=len(experts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.02)
    valid = ~batch["padding"]
    retained = batch["anchors"].exp()
    history = []
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["features"], batch["anchors"], batch["padding"], batch["domains"])
        classification = functional.cross_entropy(
            logits.reshape(-1, 40),
            batch["targets"].reshape(-1),
            ignore_index=-100,
            label_smoothing=0.03,
        )
        retention = functional.kl_div(
            functional.log_softmax(logits[valid], dim=1),
            retained[valid],
            reduction="batchmean",
        )
        loss = classification + retention_weight * retention
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step in (0, steps - 1) or (step + 1) % 100 == 0:
            with torch.no_grad():
                correct = int((logits.argmax(2)[valid] == batch["targets"][valid]).sum())
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "classification": float(classification.detach()),
                    "retention": float(retention.detach()),
                    "train_correct": correct,
                }
            )
    return model.eval(), history


@torch.no_grad()
def predict(model, experts, indices, domains, presence, records):
    batch = make_batch(experts, indices, domains, presence, records)
    model.eval()
    logits = model(batch["features"], batch["anchors"], batch["padding"], batch["domains"])
    probabilities = logits.softmax(dim=2).numpy()
    output = np.empty((batch["rows"], 40), dtype=np.float32)
    for sequence_position, row_positions in enumerate(batch["output_positions"]):
        output[row_positions] = probabilities[sequence_position, : len(row_positions)]
    return output, {"rows": batch["rows"], "rounds": batch["rounds"]}


def save_bundle(model, path, retention_weight, steps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().half() for name, value in model.state_dict().items()}
    torch.save(
        {
            "format": "supervised_expert_round_transformer_v1",
            "state_dict": state,
            "retention_weight": retention_weight,
            "steps": steps,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "inputs": ["V52", "V24 adapted", "V24 base", "X3D-L", "skeleton", "IMU"],
        },
        path,
    )


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("format") != "supervised_expert_round_transformer_v1":
        raise ValueError("Unexpected supervised expert Transformer format")
    model = ExpertRoundTransformer(expert_count=len(bundle["inputs"]))
    model.load_state_dict(
        {name: value.float() for name, value in bundle["state_dict"].items()},
        strict=True,
    )
    return model.eval(), bundle
