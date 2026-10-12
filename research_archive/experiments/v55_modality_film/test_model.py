#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.r2plus1d_models import VideoEncoder


def main():
    torch.manual_seed(42)
    reference = VideoEncoder(input_channels=3).eval()
    candidate = VideoEncoder(input_channels=3, modality_film=True).eval()
    missing, unexpected = candidate.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected
    assert missing == [f"modality_affines.{index}" for index in range(5)]
    assert sum(parameter.numel() for parameter in candidate.modality_affines) == 6144
    videos = [torch.randn(1, 3, 4, 32, 32) for _ in range(3)]
    with torch.no_grad():
        expected = reference.forward_joint(videos)
        actual = candidate.forward_joint(videos)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    print("V55 zero-initialization equivalence passed; added parameters: 6144")


if __name__ == "__main__":
    main()
