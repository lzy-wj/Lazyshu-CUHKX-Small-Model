import hashlib
import json

import numpy as np

from src.data import ROOT, get_indices


def metadata_hash():
    return hashlib.sha256((ROOT / "data/metadata/samples.json").read_bytes()).hexdigest()


def make_diagnostic_split(records, fold=0, fraction=0.2, seed=42):
    if fold < 0 or not 0 < fraction < 0.5:
        raise ValueError("Diagnostic split requires a validation fold and a fraction below 0.5")
    pool, outer = get_indices(records, fold)
    generator = np.random.default_rng(seed)
    inner = []
    for label in range(40):
        candidates = np.asarray([index for index in pool if records[index]["label"] == label])
        if len(candidates) < 2:
            raise ValueError("Every class needs at least two training-pool clips")
        generator.shuffle(candidates)
        count = min(len(candidates) - 1, max(1, round(len(candidates) * fraction)))
        inner.extend(candidates[:count].tolist())
    inner = sorted(inner)
    held_out = set(inner)
    fit = [index for index in pool if index not in held_out]
    result = {"fold": fold, "fraction": fraction, "seed": seed, "metadata_sha256": metadata_hash(),
              "train": fit, "valid": inner, "outer_valid": outer,
              "unit": "whole multimodal clip; label-stratified holdout inside the existing training-user pool",
              "selection_partition": "valid (same-user held-out clips); outer_valid is diagnostic only"}
    validate_diagnostic_split(result, records, fold)
    return result


def validate_diagnostic_split(split, records, fold):
    if fold < 0 or split["fold"] != fold or split["metadata_sha256"] != metadata_hash():
        raise ValueError("Diagnostic fold or metadata mismatch")
    pool, expected_outer = get_indices(records, fold)
    fit, inner, outer = [split[name] for name in ["train", "valid", "outer_valid"]]
    if any(not part or len(part) != len(set(part)) for part in [fit, inner, outer]):
        raise ValueError("Partitions must be nonempty with unique clip indices")
    if set(fit) & set(inner) or set(fit) | set(inner) != set(pool) or outer != expected_outer:
        raise ValueError("Partitions must preserve the original user-isolated outer holdout")
    if any(records[index]["split"] != "train" for index in fit + inner + outer):
        raise ValueError("Official test clips cannot enter diagnostic partitions")
    fit_users = {records[index]["user"] for index in fit}
    inner_users = {records[index]["user"] for index in inner}
    outer_users = {records[index]["user"] for index in outer}
    grouped = split.get("grouped_users", False)
    if grouped and (fit_users & inner_users or (fit_users | inner_users) & outer_users):
        raise ValueError("Grouped train, inner and outer users must be disjoint")
    if not grouped and (not inner_users <= fit_users or (fit_users | inner_users) & outer_users):
        raise ValueError("Inner users must be seen in training and outer users must remain unseen")
    if any({records[index]["label"] for index in part} != set(range(40)) for part in [fit, inner]):
        raise ValueError("Fit and same-user holdout must both cover all 40 classes")
    folders = []
    for part in [fit, inner, outer]:
        folders.append({(name, path) for index in part for name, path in records[index]["modalities"].items()})
    if folders[0] & folders[1] or folders[0] & folders[2] or folders[1] & folders[2]:
        raise ValueError("The same raw modality folder cannot occur in two partitions")
    return fit, inner, outer


def make_grouped_split(records, fold, seed=20260907):
    pool, outer = get_indices(records, fold)
    users = sorted({records[index]["user"] for index in pool})
    generator = np.random.default_rng(seed + fold)
    for attempt in range(20000):
        held_users = set(generator.choice(users, size=4, replace=False).tolist())
        inner = [index for index in pool if records[index]["user"] in held_users]
        fit = [index for index in pool if records[index]["user"] not in held_users]
        if not 0.15 <= len(inner) / len(pool) <= 0.30:
            continue
        if any({records[index]["label"] for index in part} != set(range(40)) for part in (fit, inner)):
            continue
        split = {"fold": fold, "seed": seed, "grouped_users": True, "metadata_sha256": metadata_hash(),
                 "train": fit, "valid": inner, "outer_valid": outer,
                 "selection_partition": "inner unseen-user holdout; outer evaluated once after training",
                 "generation": "First seeded four-user draw with all classes in fit/inner and 15-30% inner clips",
                 "attempt": attempt + 1}
        validate_diagnostic_split(split, records, fold)
        return split
    raise RuntimeError("No class-covered grouped split found; do not silently fall back to clip splitting")


def load_diagnostic_split(path, records, fold):
    return validate_diagnostic_split(json.loads(path.read_text()), records, fold)
