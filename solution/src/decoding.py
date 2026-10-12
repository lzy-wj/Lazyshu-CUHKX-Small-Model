import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.runtime_data import acquisition_date, record_interval


def rounds(indices, records, gap_seconds=30, radar_only=False):
    values = [
        (record_interval(records[int(index)], radar_only=radar_only), int(index))
        for index in indices
    ]
    values = [value for value in values if value[0] is not None]
    values.sort()
    output = []
    for interval, index in values:
        if not output or interval[0] - output[-1][-1][0][1] > gap_seconds:
            output.append([])
        output[-1].append((interval, index))
    return [
        {
            "rows": [index for _, index in group],
            "start": group[0][0][0],
            "end": max(interval[1] for interval, _ in group),
        }
        for group in output
    ]


def probability_similarity(left, right, probability_by_index):
    if len(left["rows"]) != len(right["rows"]):
        return -1.0
    first = np.stack([probability_by_index[index] for index in left["rows"]])
    second = np.stack([probability_by_index[index] for index in right["rows"]])
    first = np.sqrt(np.clip(first, 1e-12, 1))
    second = np.sqrt(np.clip(second, 1e-12, 1))
    return float((first * second).sum(axis=1).mean())


def pool_repeat_probabilities(probabilities, power=1.0):
    weights = np.ones((*probabilities.shape[:2], 1), dtype=np.float64)
    weights = weights / weights.sum(axis=0, keepdims=True)
    if power == 1:
        return np.sum(weights * probabilities, axis=0)
    pooled = np.sum(weights * np.power(probabilities, power), axis=0) ** (1 / power)
    return pooled / pooled.sum(axis=1, keepdims=True)


def repeat_groups(round_values, probability_by_index, maximum_gap, minimum_similarity):
    output = []
    position = 0
    while position < len(round_values):
        candidate = round_values[position : position + 3]
        if len(candidate) == 3:
            gaps = [candidate[i + 1]["start"] - candidate[i]["end"] for i in range(2)]
            similarities = [
                probability_similarity(candidate[0], candidate[i], probability_by_index)
                for i in (1, 2)
            ]
            if max(gaps) <= maximum_gap and min(similarities) >= minimum_similarity:
                output.append(candidate)
                position += 3
                continue
        candidate = round_values[position : position + 2]
        if len(candidate) == 2:
            gap = candidate[1]["start"] - candidate[0]["end"]
            similarity = probability_similarity(candidate[0], candidate[1], probability_by_index)
            if gap <= maximum_gap and similarity >= minimum_similarity:
                output.append(candidate)
                position += 2
                continue
        position += 1
    return output


def prepool(probabilities, indices, records, maximum_gap, minimum_similarity, power, radar_only=False):
    probability_by_index = {
        int(index): probability.copy() for index, probability in zip(indices, probabilities)
    }
    round_values = rounds(indices, records, radar_only=radar_only)
    groups = repeat_groups(round_values, probability_by_index, maximum_gap, minimum_similarity)
    for group in groups:
        pooled = pool_repeat_probabilities(
            np.stack(
                [[probability_by_index[index] for index in item["rows"]] for item in group]
            ),
            power,
        )
        for item in group:
            for position, index in enumerate(item["rows"]):
                probability_by_index[index] = pooled[position]
    return np.stack([probability_by_index[int(index)] for index in indices]), len(groups)


def repeat_pool(probabilities, indices, records, radar_only=False):
    probability_by_index = {
        int(index): probability.copy() for index, probability in zip(indices, probabilities)
    }
    round_values = rounds(indices, records, radar_only=radar_only)
    groups = repeat_groups(round_values, probability_by_index, 120, 0.7)
    for group in groups:
        if len({len(item["rows"]) for item in group}) != 1:
            continue
        pooled = pool_repeat_probabilities(
            np.stack(
                [[probability_by_index[index] for index in item["rows"]] for item in group]
            ),
            power=1.5,
        )
        for item in group:
            for position, index in enumerate(item["rows"]):
                probability_by_index[index] = pooled[position]
    return probability_by_index, round_values, groups


def assign_transition_unique(probabilities, transition, strength, beam_size=100):
    unary = np.log(np.clip(probabilities, 1e-12, 1))
    beams = [(0.0, (), 0)]
    for position in range(len(probabilities)):
        candidates = []
        for score, sequence, used in beams:
            values = unary[position] if position == 0 else unary[position] + strength * transition[sequence[-1]]
            for label in np.argsort(values)[::-1]:
                label = int(label)
                if not (used >> label) & 1:
                    candidates.append(
                        (score + values[label], sequence + (label,), used | (1 << label))
                    )
        candidates.sort(key=lambda value: value[0], reverse=True)
        beams = candidates[:beam_size]
    assignment = np.asarray(beams[0][1], dtype=np.int64)
    constrained = np.full_like(probabilities, 1e-12)
    constrained[np.arange(len(probabilities)), assignment] = 1
    return constrained


def model_score(sequence, unary, transition, transition_strength):
    positions = np.arange(len(sequence))
    score = float(unary[positions, sequence].sum())
    if len(sequence) > 1:
        score += transition_strength * sum(
            transition[left, right] for left, right in zip(sequence, sequence[1:])
        )
    return score


def flexible_assignment(
    probabilities,
    template,
    transition,
    transition_strength,
    template_strength,
    gap_penalty,
    maximum_edits,
    beam_size=80,
):
    unary = np.log(np.clip(probabilities, 1e-12, 1.0))
    target_length = len(probabilities)
    source_length = len(template)
    if abs(target_length - source_length) > maximum_edits:
        return None
    beams = [(0.0, 0.0, (), 0, 0, 0, 0)]
    for position in range(target_length):
        candidates = []
        for augmented, likelihood, sequence, used, source_position, matched, edits in beams:
            remaining_budget = maximum_edits - edits
            maximum_delete = min(remaining_budget, source_length - source_position)
            for deleted in range(maximum_delete + 1):
                aligned_source = source_position + deleted
                deletion_cost = gap_penalty * deleted
                if aligned_source < source_length:
                    label = int(template[aligned_source])
                    if not (used >> label) & 1:
                        transition_value = (
                            0.0
                            if not sequence
                            else transition_strength * transition[sequence[-1], label]
                        )
                        value = float(unary[position, label] + transition_value)
                        candidates.append(
                            (
                                augmented + value + template_strength - deletion_cost,
                                likelihood + value,
                                sequence + (label,),
                                used | (1 << label),
                                aligned_source + 1,
                                matched + 1,
                                edits + deleted,
                            )
                        )
                if edits + deleted < maximum_edits:
                    values = unary[position].copy()
                    if sequence:
                        values += transition_strength * transition[sequence[-1]]
                    for label in np.argsort(values)[::-1][:12]:
                        label = int(label)
                        if (used >> label) & 1:
                            continue
                        value = float(values[label])
                        candidates.append(
                            (
                                augmented + value - deletion_cost - gap_penalty,
                                likelihood + value,
                                sequence + (label,),
                                used | (1 << label),
                                aligned_source,
                                matched,
                                edits + deleted + 1,
                            )
                        )
        candidates.sort(key=lambda item: item[0], reverse=True)
        beams = candidates[:beam_size]
        if not beams:
            return None
    completed = []
    minimum_matches = max(1, math.ceil(min(target_length, source_length) * 0.6))
    for augmented, likelihood, sequence, used, source_position, matched, edits in beams:
        total_edits = edits + source_length - source_position
        if total_edits <= maximum_edits and matched >= minimum_matches:
            completed.append(
                (
                    augmented - gap_penalty * (source_length - source_position),
                    likelihood,
                    sequence,
                    matched,
                    total_edits,
                )
            )
    return max(completed, default=None, key=lambda item: item[0])


def pair_decode(probabilities, indices, records, state, v71_prepool=False, radar_only=False):
    probabilities = np.asarray(probabilities)
    if v71_prepool:
        settings = state["v71_prepool"]
        probabilities, _ = prepool(
            probabilities,
            indices,
            records,
            settings["maximum_gap"],
            settings["minimum_similarity"],
            settings["power"],
            radar_only=radar_only,
        )
    probability_by_index, round_values, _ = repeat_pool(
        probabilities, indices, records, radar_only=radar_only
    )
    decoded = {
        int(index): probability.copy() for index, probability in zip(indices, probabilities)
    }
    transition_key = "radar_transitions" if radar_only else "transitions"
    template_key = "radar_templates" if radar_only else "templates"
    transitions = {
        int(key): np.asarray(value, dtype=np.float64)
        for key, value in state[transition_key].items()
    }
    templates = {
        int(key): [tuple(sequence) for sequence in values]
        for key, values in state[template_key].items()
    }
    settings = state["decoder"]
    for item in round_values:
        date = acquisition_date(records[item["rows"][0]])
        environment = int(date is not None and date >= "2025-06-09")
        transition = transitions[environment]
        values = np.stack([probability_by_index[index] for index in item["rows"]])
        free = assign_transition_unique(
            values,
            transition,
            settings["transition_strength"],
            beam_size=settings["assignment_beam_size"],
        ).argmax(axis=1)
        unary = np.log(np.clip(values, 1e-12, 1.0))
        free_score = model_score(
            free, unary, transition, settings["transition_strength"]
        )
        selected = tuple(int(value) for value in free)
        source = state["source_by_date"].get(date)
        best = (free_score, free_score, selected, 0, 0)
        if source is not None:
            for template in templates[int(source)]:
                candidate = flexible_assignment(
                    values,
                    template,
                    transition,
                    settings["transition_strength"],
                    settings["template_strength"],
                    settings["gap_penalty"],
                    settings["maximum_edits"],
                    beam_size=settings["alignment_beam_size"],
                )
                if candidate is not None and candidate[0] > best[0]:
                    best = candidate
        if best[2] != selected:
            selected = best[2]
        constrained = np.full_like(values, 1e-12)
        constrained[np.arange(len(selected)), selected] = 1.0
        for position, index in enumerate(item["rows"]):
            decoded[index] = constrained[position]
    return np.stack([decoded[int(index)] for index in indices])


class DomainLogitResidualConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Conv1d(80, 40, kernel_size=1)

    def forward(self, adapted_log_probabilities, base_log_probabilities):
        features = torch.cat([adapted_log_probabilities, base_log_probabilities], dim=1)
        correction = self.residual(features.unsqueeze(-1)).squeeze(-1)
        return adapted_log_probabilities + correction


def normalized_logs(probabilities):
    values = np.asarray(probabilities, dtype=np.float32)
    values = np.clip(values, 1e-7, None)
    values /= values.sum(axis=1, keepdims=True)
    return np.log(values)


def apply_v52(path, adapted, base, groups):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    output = adapted.copy()
    for group, state in payload["state_dicts"].items():
        selected = np.asarray(groups) == group
        if not selected.any():
            continue
        model = DomainLogitResidualConv()
        model.load_state_dict(
            {name: value.float() for name, value in state.items()}, strict=True
        )
        with torch.no_grad():
            logits = model(
                torch.from_numpy(normalized_logs(adapted[selected])),
                torch.from_numpy(normalized_logs(base[selected])),
            )
            output[selected] = logits.softmax(dim=1).numpy()
    weights = {
        "unmatched_env0": 1.0,
        "paired_env0": 1.0,
        "paired_env1": 1.0,
        "unmatched_env1": 0.5,
    }
    return np.stack(
        [
            (1 - weights[group]) * base[position] + weights[group] * output[position]
            for position, group in enumerate(groups)
        ]
    )


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
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(dimension)
        self.residual = nn.Linear(dimension, 40)

    def forward(self, features, anchor_logs, padding_mask, domains):
        tokens = self.input_projection(features)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        tokens = tokens + self.domain_embedding(domains).unsqueeze(1)
        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return anchor_logs + self.residual(self.normalization(tokens))


def make_v57_batch(anchor, adapted, base, indices, domains, records):
    positions = {int(index): position for position, index in enumerate(indices)}
    sequences = rounds(indices, records)
    covered = {index for item in sequences for index in item["rows"]}
    sequences.extend(
        {"rows": [int(index)], "start": float("inf"), "end": float("inf")}
        for index in indices
        if int(index) not in covered
    )
    maximum_length = max(len(item["rows"]) for item in sequences)
    feature_values = np.concatenate(
        [normalized_logs(anchor), normalized_logs(adapted), normalized_logs(base)], axis=1
    )
    anchor_values = normalized_logs(anchor)
    features = np.zeros((len(sequences), maximum_length, 120), dtype=np.float32)
    anchors = np.zeros((len(sequences), maximum_length, 40), dtype=np.float32)
    padding = np.ones((len(sequences), maximum_length), dtype=bool)
    domain_values = np.zeros(len(sequences), dtype=np.int64)
    output_positions = []
    for sequence_position, item in enumerate(sequences):
        row_positions = np.asarray([positions[int(index)] for index in item["rows"]])
        length = len(row_positions)
        features[sequence_position, :length] = feature_values[row_positions]
        anchors[sequence_position, :length] = anchor_values[row_positions]
        padding[sequence_position, :length] = False
        domain_values[sequence_position] = int(domains[row_positions[0]])
        output_positions.append(row_positions)
    return features, anchors, padding, domain_values, output_positions


def apply_v57(path, anchor, adapted, base, indices, domains, records):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = RoundResidualTransformer()
    model.load_state_dict(
        {name: value.float() for name, value in payload["state_dict"].items()}, strict=True
    )
    batch = make_v57_batch(anchor, adapted, base, indices, domains, records)
    with torch.no_grad():
        probabilities = model(
            torch.from_numpy(batch[0]),
            torch.from_numpy(batch[1]),
            torch.from_numpy(batch[2]),
            torch.from_numpy(batch[3]),
        ).softmax(dim=2).numpy()
    output = np.empty((len(indices), 40), dtype=np.float32)
    for sequence_position, row_positions in enumerate(batch[4]):
        output[row_positions] = probabilities[sequence_position, : len(row_positions)]
    return output


class SoftRoundResidualTransformer(nn.Module):
    def __init__(
        self,
        expert_count=4,
        presence_dimension=6,
        dimension=256,
        maximum_length=16,
        layers=3,
        active_domain=2,
    ):
        super().__init__()
        self.input_projection = nn.Linear(expert_count * 40 + presence_dimension, dimension)
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
        self.active_domain = active_domain

    def forward(self, features, anchor_logs, padding_mask, domains):
        values = self.input_projection(features)
        values = values + self.position_embedding[:, : values.shape[1]]
        values = values + self.domain_embedding(domains).unsqueeze(1)
        values = self.encoder(values, src_key_padding_mask=padding_mask)
        logits = anchor_logs + self.residual(self.normalization(values))
        gate = (domains == self.active_domain).to(logits.dtype).view(-1, 1, 1)
        return anchor_logs + gate * (logits - anchor_logs)


def make_v71_batch(experts, indices, domains, presence, records):
    positions = {int(index): position for position, index in enumerate(indices)}
    sequences = rounds(indices, records)
    covered = {index for item in sequences for index in item["rows"]}
    sequences.extend(
        {"rows": [int(index)], "start": float("inf"), "end": float("inf")}
        for index in indices
        if int(index) not in covered
    )
    maximum_length = max(len(item["rows"]) for item in sequences)
    feature_values = np.concatenate(
        [normalized_logs(probabilities) for probabilities in experts]
        + [2 * np.asarray(presence, dtype=np.float32) - 1],
        axis=1,
    )
    anchor_values = normalized_logs(experts[0])
    features = np.zeros(
        (len(sequences), maximum_length, feature_values.shape[1]), dtype=np.float32
    )
    anchors = np.zeros((len(sequences), maximum_length, 40), dtype=np.float32)
    padding = np.ones((len(sequences), maximum_length), dtype=bool)
    domain_values = np.zeros(len(sequences), dtype=np.int64)
    output_positions = []
    for sequence_position, item in enumerate(sequences):
        row_positions = np.asarray([positions[int(index)] for index in item["rows"]])
        length = len(row_positions)
        features[sequence_position, :length] = feature_values[row_positions]
        anchors[sequence_position, :length] = anchor_values[row_positions]
        padding[sequence_position, :length] = False
        domain_values[sequence_position] = int(domains[row_positions[0]])
        output_positions.append(row_positions)
    return features, anchors, padding, domain_values, output_positions


def apply_v71(path, experts, indices, domains, presence, records):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SoftRoundResidualTransformer(
        expert_count=len(payload["inputs"]),
        dimension=payload["dimension"],
        layers=payload["layers"],
        active_domain=payload["active_domain"],
    )
    model.load_state_dict(
        {name: value.float() for name, value in payload["state_dict"].items()}, strict=True
    )
    batch = make_v71_batch(experts, indices, domains, presence, records)
    with torch.no_grad():
        probabilities = model(
            torch.from_numpy(batch[0]),
            torch.from_numpy(batch[1]),
            torch.from_numpy(batch[2]),
            torch.from_numpy(batch[3]),
        ).softmax(dim=2).numpy()
    output = np.empty((len(indices), 40), dtype=np.float32)
    for sequence_position, row_positions in enumerate(batch[4]):
        output[row_positions] = probabilities[sequence_position, : len(row_positions)]
    return output


def ranks(probabilities):
    output = np.empty_like(probabilities, dtype=np.int64)
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    output[np.arange(len(probabilities))[:, None], order] = np.arange(
        1, probabilities.shape[1] + 1
    )
    return output
