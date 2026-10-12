from pathlib import Path
import csv
import re
import sys


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PARTS = {
    "data",
    "outputs",
    "checkpoints",
    "swanlog",
    "declarations",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pem", ".key"}
FORBIDDEN_FILENAMES = {
    "decoder_state.json",
    "final_submission_1.csv",
    "final_submission_2.csv",
    "submission_info.yaml",
}
REQUIRED_FILES = {
    "README.md",
    "README_en.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "docs/MODEL_CARD.md",
    "docs/REPRODUCIBILITY.md",
    "docs/technical_report.pdf",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
    "credential": re.compile(
        r"(?i)(?:api[_-]?key|token|password)\s*[:=]\s*[\"'][^\"']{12,}[\"']"
    ),
}


def fail(messages):
    for message in messages:
        print(f"ERROR: {message}")
    raise SystemExit(1)


def main():
    errors = []
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    relative_files = {path.relative_to(ROOT).as_posix() for path in files}

    for required in sorted(REQUIRED_FILES - relative_files):
        errors.append(f"missing required file: {required}")

    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            errors.append(f"forbidden path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary or credential: {relative}")
        if path.name in FORBIDDEN_FILENAMES:
            errors.append(f"private model or submission artifact: {relative}")
        if path.stat().st_size > 90_000_000:
            errors.append(f"file exceeds 90 MB: {relative}")
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        private_home = "/data" + "/home/lzy"
        private_host = "trio-" + "dev02"
        if private_home in text or private_host in text:
            errors.append(f"unsanitized server metadata: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label}: {relative}")

    run_index = ROOT / "docs/swanlab_run_index.csv"
    with run_index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 596:
        errors.append(f"expected 596 SwanLab runs, found {len(rows)}")
    if len({row["run_id"] for row in rows}) != len(rows):
        errors.append("duplicate SwanLab run IDs")

    if errors:
        fail(errors)
    print(f"Repository validation passed: {len(files)} files, {len(rows)} SwanLab runs.")


if __name__ == "__main__":
    main()
