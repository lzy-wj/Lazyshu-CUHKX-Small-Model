import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    selected = json.loads((ROOT / "configs/v4_selected_ensemble.json").read_text())
    sources = [ROOT / "outputs/checkpoints" / name / "best.pt" for name in selected["final_models"]]
    if not all(path.exists() for path in sources):
        raise RuntimeError("Selected full-data checkpoints are not complete")
    common = ["--weights", *map(str, selected["weights"]), "--temperatures", *map(str, selected["temperatures"])]
    subprocess.run([sys.executable, "scripts/predict.py", "--checkpoints", *map(str, sources),
                    *common, "--name", "research_multimodal_v4"], cwd=ROOT, check=True)
    folder = ROOT / "outputs/final_v4/models"
    folder.mkdir(parents=True, exist_ok=True)
    packaged = []
    for name, source in zip(selected["final_models"], sources):
        target = folder / f"{name}.pt"
        shutil.copy2(source, target)
        packaged.append(target)
    subprocess.run([sys.executable, "scripts/predict.py", "--checkpoints", *map(str, packaged),
                    *common, "--name", "research_multimodal_v4_recheck"], cwd=ROOT, check=True)
    previous = json.loads((ROOT / "configs/v3_selected_ensemble.json").read_text())
    old_models = [ROOT / "outputs/final_v3/models" / f"{name}.pt" for name in previous["final_models"]]
    subprocess.run([sys.executable, "scripts/predict.py", "--checkpoints", *map(str, old_models),
                    "--weights", *map(str, previous["weights"]), "--temperatures", *map(str, previous["temperatures"]),
                    "--name", "research_multimodal_v3_v4_regression"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/finalize_v4.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
