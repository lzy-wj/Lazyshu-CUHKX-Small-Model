import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Kevin-Pal/CUHK-X_Small_Model_Track"
DESTINATION = ROOT / "data/downloads/CUHK-X_Small_Model_Track"
MANIFEST = ROOT / "data/metadata/download_manifest.json"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verify(manifest):
    results = []
    for entry in manifest["files"]:
        path = DESTINATION / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise RuntimeError(f"Missing or wrong size: {entry['path']}")
        sha256 = hashlib.sha256()
        git_hash = hashlib.sha1(f"blob {entry['size']}\0".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                sha256.update(chunk)
                git_hash.update(chunk)
        digest = sha256.hexdigest()
        if entry.get("sha256") and digest != entry["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {entry['path']}")
        if entry.get("git_blob") and git_hash.hexdigest() != entry["git_blob"]:
            raise RuntimeError(f"Git blob mismatch: {entry['path']}")
        results.append({"path": entry["path"], "size": entry["size"], "sha256": digest,
                        "upstream_hash_verified": bool(entry.get("sha256") or entry.get("git_blob"))})
        print(f"Verified {entry['path']}", flush=True)
    write_json(ROOT / "data/metadata/verification.json", {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "revision": manifest["revision"], "files": results,
    })


def main():
    parser = argparse.ArgumentParser(description="Download and verify the gated CUHK-X dataset.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.verify_only:
        verify(json.loads(MANIFEST.read_text()))
        return
    endpoint = os.environ["HF_ENDPOINT"].rstrip("/")
    if endpoint not in ("https://huggingface.co", "https://hf-mirror.com"):
        raise RuntimeError("HF_ENDPOINT must be https://huggingface.co or https://hf-mirror.com")
    if not get_token():
        raise RuntimeError(
            "Dataset access requires authorization. Accept access conditions on its Hugging Face page, "
            "then authenticate locally with: HF_ENDPOINT=https://huggingface.co "
            "conda run --no-capture-output -n Train hf auth login. Do not put tokens in chat or files here."
        )
    probe_path = "Small-Model-Track/Testing/data/small_model_track_test.zip"
    with requests.get(
        f"{endpoint}/datasets/{REPO_ID}/resolve/main/{probe_path}",
        headers={"Authorization": f"Bearer {get_token()}"},
        allow_redirects=False, stream=True, timeout=(10, 30),
    ) as response:
        redirect_host = urlsplit(response.headers.get("Location", "")).hostname
        if endpoint == "https://hf-mirror.com" and response.is_redirect and redirect_host == "huggingface.co":
            raise RuntimeError(
                "hf-mirror redirects this dataset to huggingface.co; stopping rather than "
                "silently downloading from the origin. HF login alone also does not grant gated "
                "dataset access. Request access on the dataset page, or use the official Baidu "
                "share recorded in README.md for domestic downloading."
            )
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Dataset access denied (HTTP {response.status_code}). Login is not dataset authorization; "
                "request access on the dataset page and check the token's read permissions."
            )
        response.raise_for_status()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=endpoint)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        manifest["endpoint"] = endpoint
        write_json(MANIFEST, manifest)
    else:
        info = api.dataset_info(REPO_ID, files_metadata=True, timeout=30)
        entries = []
        for sibling in info.siblings:
            if sibling.rfilename == ".gitattributes":
                continue
            entry = {"path": sibling.rfilename, "size": sibling.size}
            if sibling.lfs:
                checksum = sibling.lfs.sha256
                if checksum and re.fullmatch(r"[0-9a-f]{64}", checksum):
                    entry["sha256"] = checksum
            elif sibling.blob_id:
                entry["git_blob"] = sibling.blob_id
            entries.append(entry)
        manifest = {"repo_id": REPO_ID, "endpoint": endpoint, "revision": info.sha, "files": entries}
        write_json(MANIFEST, manifest)
    total = sum(entry["size"] for entry in manifest["files"])
    if shutil.disk_usage(DESTINATION).free < total:
        raise RuntimeError(f"Keep at least {total / 2**30:.2f} GiB free for this download.")
    print(f"Endpoint {endpoint}; revision {manifest['revision']}; {total / 2**30:.2f} GiB; {args.workers} workers", flush=True)
    hf_hub_download(REPO_ID, "README.md", repo_type="dataset", revision=manifest["revision"],
                    local_dir=DESTINATION, endpoint=endpoint)
    snapshot_download(REPO_ID, repo_type="dataset", revision=manifest["revision"],
                      local_dir=DESTINATION, endpoint=endpoint, max_workers=args.workers,
                      allow_patterns=[entry["path"] for entry in manifest["files"]])
    verify(manifest)
    print("Download verified. Archives remain untouched; extraction is a separate step.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        raise SystemExit(f"Download failed (HTTP {status}). Check dataset authorization/network; rerun to resume.")
    except (LocalEntryNotFoundError, requests.RequestException):
        raise SystemExit("Download request failed. Check connectivity and dataset access; rerun to resume.")
    except (RuntimeError, OSError, ValueError) as error:
        raise SystemExit(str(error))
