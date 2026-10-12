#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateSubmissionRequest,
    ApiStartSubmissionUploadRequest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--message", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.file.is_file():
        raise FileNotFoundError(args.file)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        upload_request = ApiStartSubmissionUploadRequest()
        upload_request.competition_name = args.competition
        upload_request.file_name = args.file.name
        upload_request.content_length = args.file.stat().st_size
        upload_request.last_modified_epoch_seconds = int(args.file.stat().st_mtime)
        upload = kaggle.competitions.competition_api_client.start_submission_upload(upload_request)
        upload_url = upload.create_url.replace(
            "https://www.googleapis.com/", "https://storage.googleapis.com/"
        )
        subprocess.run(
            [
                "curl", "--fail-with-body", "--silent", "--show-error",
                "--retry", "5", "--retry-all-errors", "--connect-timeout", "20",
                "--max-time", "180", "--request", "PUT", "--upload-file", str(args.file),
                upload_url,
            ],
            check=True,
        )
        request = ApiCreateSubmissionRequest()
        request.competition_name = args.competition
        request.blob_file_tokens = upload.token
        request.submission_description = args.message
        response = kaggle.competitions.competition_api_client.create_submission(request)
    print(response)


if __name__ == "__main__":
    main()
