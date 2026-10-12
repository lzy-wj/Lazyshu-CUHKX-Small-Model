from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    inputs = np.load(ROOT / "data/processed/skeleton.npy", mmap_mode="r")
    upper = np.triu_indices(17, 1)
    parents = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    output = np.lib.format.open_memmap(ROOT / "data/processed/geometry.npy", mode="w+", dtype=np.float32,
                                       shape=(len(inputs), 64, 408))
    for index, features in enumerate(inputs):
        pose = features[:, :51].reshape(64, 17, 3).copy()
        scale = np.median(np.linalg.norm(pose[:, 8] - pose[:, 0], axis=-1)).clip(0.15, 1.2)
        distances = np.linalg.norm(pose[:, upper[0]] - pose[:, upper[1]], axis=-1) / scale
        bones = pose - pose[:, parents]
        bones /= np.linalg.norm(bones, axis=-1, keepdims=True).clip(1e-6)
        angles = (bones[:, upper[0]] * bones[:, upper[1]]).sum(-1)
        velocity = np.diff(distances, axis=0, prepend=distances[:1]) * 5
        output[index] = np.concatenate([distances / 3, angles, velocity], axis=1).clip(-10, 10)
    output.flush()
    print("Prepared geometry features:", output.shape)


if __name__ == "__main__":
    main()
