import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    selected = json.loads((ROOT / "configs/v3_selected_ensemble.json").read_text())
    sources = [ROOT / "outputs/checkpoints" / name / "best.pt" for name in selected["final_models"]]
    if not all(path.exists() for path in sources):
        raise RuntimeError("All selected full-data checkpoints must exist before prediction")
    common = ["--weights", *map(str, selected["weights"]), "--temperatures", *map(str, selected["temperatures"])]
    command = [sys.executable, str(ROOT / "scripts/predict.py"), "--checkpoints", *map(str, sources),
               *common, "--name", "research_multimodal_v3"]
    subprocess.run(command, cwd=ROOT, check=True)
    folder = ROOT / "outputs/final_v3/models"
    folder.mkdir(parents=True, exist_ok=True)
    packaged = []
    for name, source in zip(selected["final_models"], sources):
        target = folder / f"{name}.pt"
        shutil.copy2(source, target)
        packaged.append(target)
    command = [sys.executable, str(ROOT / "scripts/predict.py"), "--checkpoints", *map(str, packaged),
               *common, "--name", "research_multimodal_v3_recheck"]
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/finalize_v3.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
