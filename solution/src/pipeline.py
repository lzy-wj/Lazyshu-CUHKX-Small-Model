import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.decoding import (
    apply_v52,
    apply_v57,
    apply_v71,
    pair_decode,
    ranks,
)
from src.modeling import infer_base_variant, infer_protected_variant
from src.runtime_data import acquisition_date, modality_presence


PRIMARY_SUBMISSION_ID = "56174717"
SECONDARY_SUBMISSION_ID = "56243275"


def artifact_paths(package_root):
    root = Path(package_root) / "artifacts"
    return {
        "base": root / "v24_full_warm60_soup_int8.pt",
        "low_delta": root / "v24_lowres_pseudo_head_delta.pt",
        "high_delta": root / "v24_highres_pseudo_head_delta.pt",
        "v103": root / "v103_full_protected_dual_adapter.pt",
        "v104_old": root / "v104_target_old_cross_state.pt",
        "v104_refreshed": root / "v104_target_refreshed_cross_state.pt",
        "v114_16": root / "v114_source16_state.pt",
        "v114_20": root / "v114_source20_state.pt",
        "v114_21": root / "v114_source21_state.pt",
        "v52": root / "v52_domain_logit_residual_conv1d.pt",
        "v57": root / "v57_round_residual_transformer.pt",
        "v71": root / "v71_gated_film_transformer.pt",
        "decoder": root / "decoder_state.json",
    }


def domain_values(records, state):
    dates = [acquisition_date(record) for record in records]
    sources = [state["source_by_date"].get(date) for date in dates]
    paired = np.asarray([source is not None for source in sources], dtype=bool)
    environments = np.asarray(
        [int(date is not None and date >= "2025-06-09") for date in dates],
        dtype=np.int64,
    )
    domains = environments + 2 * paired.astype(np.int64)
    groups = np.asarray(
        [
            f"{'paired' if is_paired else 'unmatched'}_env{environment}"
            for is_paired, environment in zip(paired, environments)
        ]
    )
    return dates, sources, paired, domains, groups


def compose_predictions(
    records,
    state,
    paths,
    probabilities,
    presence,
):
    indices = np.arange(len(records), dtype=np.int64)
    dates, sources, paired, domains, groups = domain_values(records, state)
    base = probabilities["base"]
    adapted_low = probabilities["adapted_low"]
    v24_decoded = pair_decode(adapted_low, indices, records, state)
    v52 = apply_v52(paths["v52"], adapted_low, base, groups)
    v57 = apply_v57(paths["v57"], v52, adapted_low, base, indices, domains, records)
    v71 = apply_v71(
        paths["v71"],
        [v57, v52, adapted_low, base],
        indices,
        domains,
        presence,
        records,
    )
    v71_decoded = pair_decode(v71, indices, records, state, v71_prepool=True)

    v89 = (0.5 * (
        adapted_low.astype(np.float64)
        + probabilities["adapted_high"].astype(np.float64)
    )).astype(np.float32)
    v104 = v89.astype(np.float64)
    unmatched_env1 = groups == "unmatched_env1"
    v104[unmatched_env1] = (
        v89[unmatched_env1].astype(np.float64)
        + probabilities["v103"][unmatched_env1].astype(np.float64)
        + probabilities["v104_old"][unmatched_env1].astype(np.float64)
        + probabilities["v104_refreshed"][unmatched_env1].astype(np.float64)
    ) / 4
    paired_env0 = groups == "paired_env0"
    v104[paired_env0] = 0.5 * (
        v89[paired_env0].astype(np.float64)
        + probabilities["v103"][paired_env0].astype(np.float64)
    )

    v114 = v104.copy()
    for source, key in ((16, "v114_16"), (20, "v114_20"), (21, "v114_21")):
        selected = np.asarray([value == source for value in sources])
        if selected.any():
            v114[selected] = (
                0.4 * v104[selected]
                + 0.6 * probabilities[key].astype(np.float64)
            )
    v114_decoded = pair_decode(v114, indices, records, state)

    v127 = np.where(paired[:, None], v114_decoded, v71_decoded).argmax(axis=1)
    source_support = {
        int(key): set(values) for key, values in state["source_support"].items()
    }
    incumbent = v24_decoded.argmax(axis=1)
    protected_v127 = np.asarray(
        [
            source is not None and int(incumbent[position]) not in source_support[int(source)]
            for position, source in enumerate(sources)
        ],
        dtype=bool,
    )
    v127[protected_v127] = incumbent[protected_v127]
    full_v127 = v127.copy()
    primary = full_v127.copy()
    rollback = np.asarray(
        [
            date == "2025-05-31" and primary[position] != incumbent[position]
            for position, date in enumerate(dates)
        ]
    )
    primary[rollback] = incumbent[rollback]

    v71_radar = pair_decode(
        v71,
        indices,
        records,
        state,
        v71_prepool=True,
        radar_only=True,
    )
    v114_radar = pair_decode(v114, indices, records, state, radar_only=True)
    challenger = np.where(paired[:, None], v114_radar, v71_radar).argmax(axis=1)
    protected_challenger = np.asarray(
        [
            source is not None
            and int(challenger[position]) not in source_support[int(source)]
            for position, source in enumerate(sources)
        ],
        dtype=bool,
    )
    challenger[protected_challenger] = incumbent[protected_challenger]
    evidence = np.where(paired[:, None], v114, v71)
    evidence_ranks = ranks(evidence)
    rows = np.arange(len(records))
    accepted = (
        (primary != challenger)
        & (evidence_ranks[rows, challenger] <= state["v164_gate"]["challenger_max_rank"])
        & (evidence_ranks[rows, primary] >= state["v164_gate"]["incumbent_min_rank"])
    )
    secondary = primary.copy()
    secondary[accepted] = challenger[accepted]
    report = {
        "rows": len(records),
        "paired_rows": int(paired.sum()),
        "primary_rollback_positions": np.flatnonzero(rollback).tolist(),
        "v164_changed_positions": np.flatnonzero(accepted).tolist(),
        "known_date_rows": int(sum(date in state["source_by_date"] for date in dates)),
        "unknown_date_rows": int(sum(date not in state["source_by_date"] for date in dates)),
    }
    return {
        PRIMARY_SUBMISSION_ID: primary,
        SECONDARY_SUBMISSION_ID: secondary,
        "full_v127": full_v127,
        "v24": incumbent,
        "report": report,
        "intermediate": {
            "v52": v52,
            "v57": v57,
            "v71": v71,
            "v104": v104,
            "v114": v114,
        },
    }


def run_models(
    package_root,
    records,
    sensors_path,
    videos,
    batch_size,
    workers,
    device,
):
    paths = artifact_paths(package_root)
    indices = np.arange(len(records), dtype=np.int64)
    started = time.monotonic()
    probabilities = {}
    probabilities["base"] = infer_base_variant(
        paths["base"],
        None,
        indices,
        records,
        sensors_path,
        videos[128],
        128,
        batch_size,
        workers,
        device,
        adabn=False,
    )
    probabilities["adapted_low"] = infer_base_variant(
        paths["base"],
        paths["low_delta"],
        indices,
        records,
        sensors_path,
        videos[128],
        128,
        batch_size,
        workers,
        device,
        adabn=True,
    )
    probabilities["adapted_high"] = infer_base_variant(
        paths["base"],
        paths["high_delta"],
        indices,
        records,
        sensors_path,
        videos[160],
        160,
        batch_size,
        workers,
        device,
        adabn=True,
    )
    probabilities["v103"] = infer_protected_variant(
        paths["base"],
        paths["v103"],
        "adapter",
        indices,
        records,
        sensors_path,
        videos[160],
        batch_size,
        workers,
        device,
        tta_hflip=False,
        crop_scale=1.0,
    )
    for name, key in (("v104_old", "v104_old"), ("v104_refreshed", "v104_refreshed")):
        probabilities[name] = infer_protected_variant(
            paths["base"],
            paths[key],
            "state",
            indices,
            records,
            sensors_path,
            videos[160],
            batch_size,
            workers,
            device,
            tta_hflip=True,
            crop_scale=0.75,
        )
    state = json.loads(paths["decoder"].read_text())
    _, sources, _, _, _ = domain_values(records, state)
    for source in (16, 20, 21):
        selected = np.asarray([value == source for value in sources])
        source_indices = indices[selected]
        probabilities[f"v114_{source}"] = infer_protected_variant(
            paths["base"],
            paths[f"v114_{source}"],
            "state",
            source_indices,
            records,
            sensors_path,
            videos[160],
            batch_size,
            workers,
            device,
            tta_hflip=True,
            crop_scale=0.75,
        ) if len(source_indices) else np.empty((0, 40), dtype=np.float32)
    return probabilities, time.monotonic() - started


def run_pipeline(
    package_root,
    records,
    test_csv,
    sensors_path,
    videos,
    output_csv,
    submission_id,
    batch_size,
    workers,
    device,
    work_dir,
):
    paths = artifact_paths(package_root)
    state = json.loads(paths["decoder"].read_text())
    probabilities, model_seconds = run_models(
        package_root,
        records,
        sensors_path,
        videos,
        batch_size,
        workers,
        device,
    )
    presence = modality_presence(sensors_path)
    result = compose_predictions(records, state, paths, probabilities, presence)
    target = PRIMARY_SUBMISSION_ID if submission_id == "primary" else str(submission_id)
    if target not in (PRIMARY_SUBMISSION_ID, SECONDARY_SUBMISSION_ID):
        raise ValueError(f"Unsupported Selected Submission ID: {submission_id}")
    predicted = pd.DataFrame(
        {
            "path": [record["id"] for record in records],
            "prediction": result[target].astype(np.int64),
        }
    )
    if test_csv is not None:
        order = pd.read_csv(test_csv)[["path"]]
        if set(order["path"]) != set(predicted["path"]):
            raise ValueError("Discovered sample paths do not match supplied test.csv")
        predicted = order.merge(predicted, on="path", how="left")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(output_csv, index=False)
    work_dir = Path(work_dir)
    np.savez_compressed(
        work_dir / "probabilities.npz",
        **{name: value for name, value in probabilities.items()},
    )
    report = {
        **result["report"],
        "submission_id": target,
        "output_csv": str(output_csv),
        "model_inference_seconds": model_seconds,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }
    (work_dir / "inference_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
