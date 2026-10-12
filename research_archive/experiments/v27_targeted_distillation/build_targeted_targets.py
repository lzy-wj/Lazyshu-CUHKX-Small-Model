#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    teacher = np.load(args.teacher)
    student = np.load(args.student)
    if not np.array_equal(teacher["indices"], student["indices"]):
        raise ValueError("Teacher and student rows are not aligned")
    pseudo_labels = teacher["pseudo_labels"].astype(np.int64)
    student_labels = student["probabilities"].argmax(axis=1)
    selected = pseudo_labels != student_labels
    labels = student["labels"]
    report = {
        "rows": len(pseudo_labels),
        "selected_disagreements": int(selected.sum()),
        "student_correct_on_selected_audit_only": int(
            (student_labels[selected] == labels[selected]).sum()
        ),
        "teacher_correct_on_selected_audit_only": int(
            (pseudo_labels[selected] == labels[selected]).sum()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        probabilities=teacher["probabilities"],
        pseudo_labels=pseudo_labels,
        confidence=np.zeros(len(pseudo_labels), dtype=np.float32),
        paired=selected,
        indices=teacher["indices"],
        labels=labels,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

