#!/usr/bin/env python3
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.build_pair_aware_submission import SOURCE_BY_DATE
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


MODELS = {
    "V13": {
        "report": "experiments/scene_context_v13/outputs/shared6_mixup_true_domain_pairaware.json",
        "report_kind": "baseline",
        "submission": "outputs/submissions/v13_pair_aware.csv",
        "public_correct": 180,
    },
    "V14-fixed": {
        "report": "experiments/scene_context_v13/outputs/shared6_mixup_fixedsoup_uniform_l3int6_l4b0_domain_pairaware.json",
        "report_kind": "root",
        "submission": "outputs/submissions/v14_fixedsoup_pair_aware.csv",
        "public_correct": 178,
    },
    "V14-pseudo": {
        "report": "experiments/scene_context_v13/outputs/fixedsoup_pseudo_c0p6_e2_frozen_domain_pairaware.json",
        "report_kind": "root",
        "submission": "outputs/submissions/v14_fixedsoup_pseudo_pair_aware.csv",
        "public_correct": 179,
    },
    "V23": {
        "report": "experiments/v81_validation_rebuild/outputs/v23_direct_pairaware.json",
        "report_kind": "root",
        "submission": "outputs/submissions/v23_cv_warm60_soup_pair_aware.csv",
        "public_correct": 171,
    },
    "V24": {
        "report": "experiments/v81_validation_rebuild/outputs/v24_adapted_pairaware.json",
        "report_kind": "root",
        "submission": "experiments/v24_full_refit/outputs/submissions/v24_full_warm60_pseudo_pair_aware.csv",
        "public_correct": 183,
    },
    "V26": {
        "report": "experiments/v26_system_distillation/outputs/direct_pairaware.json",
        "report_kind": "root",
        "submission": "experiments/v24_full_refit/outputs/submissions/v26_system_distill_pair_aware.csv",
        "public_correct": 181,
    },
    "V29": {
        "report": "experiments/v29_crop_system_fusion/outputs/direct_pairaware.json",
        "report_kind": "root",
        "submission": "experiments/v24_full_refit/outputs/submissions/v29_crop_system_fusion_pair_aware.csv",
        "public_correct": 178,
    },
    "V59": {
        "report": "experiments/v59_teacher_sequence_hybrid/outputs/validation.json",
        "report_kind": "root",
        "submission": "experiments/v24_full_refit/outputs/submissions/v59_teacher_sequence_hybrid.csv",
        "public_correct": 180,
    },
    "V71": {
        "report": "experiments/v71_gated_film_distillation/outputs/validation.json",
        "report_kind": "nested",
        "submission": "experiments/v24_full_refit/outputs/submissions/v71_gated_film_sequence_hybrid.csv",
        "public_correct": 180,
    },
}

USER_DOMAIN = {3: 0, 5: 0, 23: 1, 22: 1, 8: 2, 7: 2, 18: 3, 17: 3}
PUBLIC_ROWS = 201
TEST_ROWS = 405


def test_domains(records):
    domains = Counter()
    dates = Counter()
    domain_by_path = {}
    for record in records:
        if record["split"] != "test":
            continue
        interval = record_interval(record)
        if interval is None:
            raise ValueError(f"Test row {record['index']} has no acquisition timestamp")
        date = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")
        if date not in SOURCE_BY_DATE:
            raise ValueError(f"Unexpected test acquisition date: {date}")
        domain = int(date >= "2025-06-09") + 2 * int(SOURCE_BY_DATE[date] is not None)
        domains[domain] += 1
        dates[date] += 1
        domain_by_path[record["id"]] = domain
    if sum(domains.values()) != TEST_ROWS:
        raise ValueError("Test-domain audit does not cover all rows")
    return domains, dates, domain_by_path


def per_user(report, kind):
    if kind == "baseline":
        folds = report["pair_aware_baseline"]["folds"]
    elif kind == "nested":
        folds = [report["fold0"]["student"], report["fold1"]["student"]]
    else:
        folds = [report["fold0"], report["fold1"]]
    result = {}
    for fold in folds:
        result.update({int(user): values for user, values in fold["per_user"].items()})
    if set(result) != set(USER_DOMAIN):
        raise ValueError(f"Unexpected validation users: {sorted(result)}")
    return result


def correlation(rows, key):
    local = np.asarray([row[key] for row in rows])
    public = np.asarray([row["public_accuracy"] for row in rows])
    return {
        "pearson": float(pearsonr(local, public).statistic),
        "spearman": float(spearmanr(local, public).statistic),
        "kendall": float(kendalltau(local, public).statistic),
    }


def current_v80_recheck(records, domain_weights):
    sources = {
        "V23-base": "experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz",
        "V80-masked": "experiments/v80_spatial_masked_encoder/outputs/v80_spatial_masked_encoder_f{fold}/validation.npz",
        "V80-encoder": "experiments/v80_spatial_masked_encoder/outputs/v80_spatial_encoder_only_f{fold}/validation.npz",
    }
    collected = {name: [] for name in sources}
    for fold in (0, 1):
        archives = {name: np.load(ROOT / pattern.format(fold=fold)) for name, pattern in sources.items()}
        indices = np.asarray(sorted(archives["V23-base"]["indices"].astype(int)))
        labels = np.asarray([records[int(index)]["label"] for index in indices])
        users = np.asarray([records[int(index)]["user"] for index in indices])
        for name, archive in archives.items():
            positions = {int(index): position for position, index in enumerate(archive["indices"])}
            probabilities = archive["probabilities"][[positions[int(index)] for index in indices]]
            collected[name].append((probabilities, labels, users))
        base = collected["V23-base"][-1][0]
        masked = collected["V80-masked"][-1][0]
        collected.setdefault("V80-blend-0.8", []).append((0.2 * base + 0.8 * masked, labels, users))
    rows = []
    for name, folds in collected.items():
        correct = np.concatenate([probabilities.argmax(axis=1) == labels for probabilities, labels, _ in folds])
        users = np.concatenate([users for _, _, users in folds])
        domain_accuracy = {}
        for domain in range(4):
            selected = np.asarray([USER_DOMAIN[int(user)] == domain for user in users])
            domain_accuracy[domain] = float(correct[selected].mean())
        relevant = np.asarray([USER_DOMAIN[int(user)] != 0 for user in users])
        rows.append(
            {
                "model": name,
                "old_oof_correct": int(correct.sum()),
                "old_oof_rows": len(correct),
                "core_correct": int(correct[relevant].sum()),
                "core_rows": int(relevant.sum()),
                "test_domain_weighted_accuracy": sum(
                    domain_weights.get(domain, 0.0) * accuracy
                    for domain, accuracy in domain_accuracy.items()
                ),
                "domain_accuracy": {str(key): value for key, value in domain_accuracy.items()},
            }
        )
    return rows


def main():
    records = get_records()
    domain_counts, date_counts, domain_by_path = test_domains(records)
    domain_weights = {domain: count / TEST_ROWS for domain, count in domain_counts.items()}
    reference = pd.read_csv(ROOT / MODELS["V24"]["submission"])
    rows = []
    for name, spec in MODELS.items():
        report = json.loads((ROOT / spec["report"]).read_text())
        users = per_user(report, spec["report_kind"])
        correct = sum(values["correct"] for values in users.values())
        count = sum(values["rows"] for values in users.values())
        domain_accuracy = {}
        for domain in range(4):
            selected = [values for user, values in users.items() if USER_DOMAIN[user] == domain]
            domain_accuracy[domain] = sum(values["correct"] for values in selected) / sum(
                values["rows"] for values in selected
            )
        domain_weighted = sum(
            domain_weights.get(domain, 0.0) * accuracy
            for domain, accuracy in domain_accuracy.items()
        )
        submission = pd.read_csv(ROOT / spec["submission"])
        if not submission["path"].equals(reference["path"]):
            raise ValueError(f"Submission paths are not aligned for {name}")
        disagreements = int((submission["prediction"] != reference["prediction"]).sum())
        changed = submission["prediction"] != reference["prediction"]
        disagreements_by_domain = Counter(
            domain_by_path[path] for path in submission.loc[changed, "path"]
        )
        public_accuracy = spec["public_correct"] / PUBLIC_ROWS
        rows.append(
            {
                "model": name,
                "oof_correct": correct,
                "oof_rows": count,
                "oof_accuracy": correct / count,
                "domain_accuracy": {str(key): value for key, value in domain_accuracy.items()},
                "test_domain_weighted_accuracy": domain_weighted,
                "public_correct": spec["public_correct"],
                "public_rows": PUBLIC_ROWS,
                "public_accuracy": public_accuracy,
                "public_minus_weighted_gap": public_accuracy - domain_weighted,
                "v24_disagreements": disagreements,
                "v24_disagreements_by_domain": {
                    str(key): value for key, value in sorted(disagreements_by_domain.items())
                },
                "v24_agreement": 1 - disagreements / TEST_ROWS,
                "public_net_correct_vs_v24": spec["public_correct"] - MODELS["V24"]["public_correct"],
            }
        )
    payload = {
        "protocol": (
            "Reweight the six test-relevant held-out users by the observed unlabeled test-domain proportions. "
            "Use agreement with public-best V24 only as a label-free guardrail, never as ground truth."
        ),
        "public_rows": PUBLIC_ROWS,
        "test_rows": TEST_ROWS,
        "test_date_counts": dict(sorted(date_counts.items())),
        "test_domain_counts": {str(key): value for key, value in sorted(domain_counts.items())},
        "test_domain_weights": {str(key): value for key, value in sorted(domain_weights.items())},
        "validation_user_domains": {str(key): value for key, value in USER_DOMAIN.items()},
        "models": rows,
        "correlation": {
            "old_oof": correlation(rows, "oof_accuracy"),
            "test_domain_weighted_oof": correlation(rows, "test_domain_weighted_accuracy"),
            "v24_agreement": correlation(rows, "v24_agreement"),
        },
        "current_v80_recheck": current_v80_recheck(records, domain_weights),
        "single_row_probes": {
            "row_0007": "public and V24 label correct",
            "row_0056": "unchanged; outside public subset or both labels wrong",
            "row_0108": "unchanged; outside public subset or both labels wrong",
        },
    }
    output = HERE / "outputs/validation_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["correlation"], indent=2))


if __name__ == "__main__":
    main()
