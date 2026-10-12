#!/usr/bin/env python3
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLUGS = (
    "v63_film_maskdistill",
    "v65_film_pervisual_aux",
)


def main():
    candidates = []
    for slug in SLUGS:
        report = json.loads(
            (ROOT / f"experiments/{slug}/outputs/fold0_evaluation.json").read_text()
        )
        candidates.append(
            {
                "slug": slug,
                "advance_fold1": report["advance_fold1"],
                "pair_correct": report["selected_blend"]["pair"]["correct"],
                "raw_correct": report["selected_blend"]["raw"]["correct"],
                "candidate_raw_correct": report["candidate"]["raw"]["correct"],
                "nll": report["selected_blend"]["raw"]["nll"],
                "weight": report["selected_blend"]["weight"],
            }
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["pair_correct"],
            item["raw_correct"],
            item["candidate_raw_correct"],
            -item["nll"],
        ),
        reverse=True,
    )
    selected = [item["slug"] for item in ranked if item["advance_fold1"]][:3]
    result = {
        "protocol": "Top three Fold-0 gate passers advance blind to Fold 1",
        "ranked": ranked,
        "selected": selected,
    }
    (HERE / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
