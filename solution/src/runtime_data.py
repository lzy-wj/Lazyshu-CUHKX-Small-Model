import csv
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


MODALITIES = ["Skeleton", "IMU", "Radar", "Depth_Color", "IR", "Thermal"]
DEVICES = ["WTLA", "WTRA", "WTC", "WTLL", "WTRL"]
PARENTS = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
FRAME_PATTERN = re.compile(r"_(\d+)(?:_Color)?\.(?:png|jpg)$", re.IGNORECASE)
TIMESTAMP_PATTERN = re.compile(r"(20\d\d-\d\d-\d\d_\d\d-\d\d-\d\d\.\d+)")
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)
VIDEO_MEMMAPS = {}
VIDEO_CONFIG = None
INTERVAL_CACHE = {}


def digest_text(values):
    value = hashlib.sha256()
    for item in values:
        value.update(str(item).encode())
        value.update(b"\0")
    return value.hexdigest()


def runtime_fingerprint(records):
    values = []
    for record in records:
        values.extend((record["id"], record["folder"]))
        for modality, folder_value in sorted(record["modalities"].items()):
            folder = Path(folder_value)
            files = sorted(path for path in folder.rglob("*") if path.is_file())
            values.extend((modality, len(files)))
            for path in files[:4] + files[-4:]:
                metadata = path.stat()
                values.extend(
                    (
                        path.relative_to(folder),
                        metadata.st_size,
                        metadata.st_mtime_ns,
                    )
                )
    return digest_text(values)


def find_test_csv(data_dir):
    candidates = [
        data_dir / "test_file/test.csv",
        data_dir / "Testing/test_file/test.csv",
        data_dir / "Small-Model-Track/Testing/test_file/test.csv",
    ]
    candidates.extend(sorted(data_dir.glob("**/test_file/test.csv")))
    for path in candidates:
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        if "path" in table.columns:
            return path, table
    return None, None


def resolve_clip(data_dir, sample_id):
    relative = Path(str(sample_id).strip("/"))
    candidates = [
        data_dir / relative,
        data_dir / "data" / relative,
        data_dir.parent / relative,
    ]
    if data_dir.name == "small_model_track_test":
        candidates.insert(0, data_dir / relative.name)
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    matches = [path for path in data_dir.glob(f"**/{relative.name}") if path.is_dir()]
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"Cannot resolve test clip {sample_id!r} below {data_dir}")


def discover_records(data_dir):
    data_dir = Path(data_dir).resolve()
    test_csv, table = find_test_csv(data_dir)
    if table is not None:
        sample_ids = table["path"].astype(str).tolist()
    else:
        roots = []
        if data_dir.name == "small_model_track_test":
            roots.append(data_dir)
        roots.extend(path for path in data_dir.glob("**/small_model_track_test") if path.is_dir())
        roots = list(dict.fromkeys(path.resolve() for path in roots))
        if len(roots) != 1:
            raise ValueError(
                "Could not identify a unique small_model_track_test directory or test.csv"
            )
        sample_ids = [
            f"small_model_track_test/{path.name}/"
            for path in sorted(roots[0].iterdir())
            if path.is_dir() and path.name.startswith("SM_test_")
        ]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Testing sample paths are empty or duplicated")
    records = []
    for index, sample_id in enumerate(sample_ids):
        folder = resolve_clip(data_dir, sample_id)
        records.append(
            {
                "id": sample_id,
                "index": index,
                "split": "test",
                "label": -1,
                "user": -1,
                "trial": folder.name,
                "folder": str(folder),
                "modalities": {
                    name: str((folder / name).resolve())
                    for name in MODALITIES
                    if (folder / name).is_dir()
                },
            }
        )
    return records, test_csv


def interpolate(values, steps=64):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros((steps, values.shape[-1]), dtype=np.float32)
    values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
    source = np.linspace(0, 1, len(values))
    target = np.linspace(0, 1, steps)
    return np.stack(
        [np.interp(target, source, values[:, channel]) for channel in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)


def rotate(vectors, quaternion):
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True).clip(1e-6)
    scalar = quaternion[:, :1]
    axis = quaternion[:, 1:]
    twice_cross = 2 * np.cross(axis, vectors)
    return vectors + scalar * twice_cross + np.cross(axis, twice_cross)


def sensor_features(record):
    skeleton = np.zeros((64, 153), dtype=np.float32)
    imu_world = np.zeros((64, 85), dtype=np.float32)
    sensor_mask = np.zeros(3, dtype=np.float32)
    paths = {name: Path(path) for name, path in record["modalities"].items()}
    if "Skeleton" in paths:
        frames = []
        for path in sorted(paths["Skeleton"].rglob("*.json")):
            people = json.loads(path.read_text())
            if isinstance(people, list) and people:
                person = max(people, key=lambda entry: np.mean(entry.get("keypoint_scores", [0])))
                points = np.asarray(person["keypoints"], dtype=np.float32)
                if points.shape != (17, 3):
                    raise ValueError(f"Unexpected skeleton shape {points.shape}: {path}")
                frames.append(points)
        if frames:
            pose = interpolate(np.asarray(frames).reshape(-1, 51)).reshape(64, 17, 3)
            bones = pose - pose[:, PARENTS]
            velocity = np.diff(pose, axis=0, prepend=pose[:1]) * 10
            skeleton = np.concatenate(
                [pose.reshape(64, -1), bones.reshape(64, -1), velocity.reshape(64, -1)],
                axis=1,
            )
            sensor_mask[0] = 1
    if "IMU" in paths:
        tables = [pd.read_csv(path) for path in sorted(paths["IMU"].glob("*.csv"))]
        if tables:
            table = pd.concat(tables, ignore_index=True)
            if not table.empty:
                for position, device in enumerate(DEVICES):
                    rows = table[table.iloc[:, 1].astype(str).str.startswith(device + "(")]
                    rows = rows.sort_values(table.columns[0]).drop_duplicates(table.columns[0])
                    if rows.empty:
                        continue
                    values = np.nan_to_num(
                        rows.iloc[:, 2:18]
                        .apply(pd.to_numeric, errors="coerce")
                        .to_numpy(dtype=np.float32)
                    )
                    acceleration = rotate(values[:, :3], values[:, 12:16])
                    gyroscope = rotate(values[:, 3:6], values[:, 12:16]) / 180
                    dynamic = acceleration - np.median(acceleration, axis=0, keepdims=True)
                    centered_gyro = gyroscope - np.median(gyroscope, axis=0, keepdims=True)
                    magnitudes = np.stack(
                        [
                            np.linalg.norm(acceleration, axis=1),
                            np.linalg.norm(gyroscope, axis=1),
                            np.linalg.norm(
                                np.diff(acceleration, axis=0, prepend=acceleration[:1]),
                                axis=1,
                            ),
                            np.ones(len(rows)),
                        ],
                        axis=1,
                    )
                    features = np.concatenate(
                        [acceleration / 4, gyroscope, dynamic, centered_gyro, magnitudes],
                        axis=1,
                    )
                    imu_world[:, position * 16 : (position + 1) * 16] = interpolate(features)
                    imu_world[:, 80 + position] = 1
                sensor_mask[1] = float(imu_world[:, 80:].any())
    if "Radar" in paths:
        tables = [pd.read_csv(path) for path in sorted(paths["Radar"].glob("*.csv"))]
        sensor_mask[2] = float(any(not table.empty for table in tables))
    video_mask = np.zeros(3, dtype=np.float32)
    for modality_index, modality in enumerate(("Depth_Color", "IR", "Thermal")):
        files = frame_files(paths.get(modality, ""))
        for frame_index in sorted(set(pick_indices(len(files), 24))):
            try:
                with Image.open(files[frame_index]) as image:
                    image.verify()
                video_mask[modality_index] = 1
                break
            except (OSError, ValueError):
                continue
    return (
        skeleton.clip(-10, 10),
        imu_world.clip(-10, 10),
        sensor_mask,
        video_mask,
    )


def prepare_sensors(records, cache_dir, workers, fingerprint):
    output = cache_dir / "sensors.npz"
    if output.is_file():
        archive = np.load(output)
        stored = str(archive["sample_fingerprint"]) if "sample_fingerprint" in archive else None
        if len(archive["skeleton"]) == len(records) and stored == fingerprint:
            return output
    with ProcessPoolExecutor(max_workers=workers) as pool:
        values = list(
            tqdm(
                pool.map(sensor_features, records, chunksize=8),
                total=len(records),
                desc="Sensors",
            )
        )
    np.savez(
        output,
        skeleton=np.stack([value[0] for value in values]),
        imu_world=np.stack([value[1] for value in values]),
        sensor_mask=np.stack([value[2] for value in values]),
        video_mask=np.stack([value[3] for value in values]),
        sample_fingerprint=np.asarray(fingerprint),
    )
    return output


def frame_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [
        path
        for path in folder.iterdir()
        if path.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]

    def key(path):
        match = FRAME_PATTERN.search(path.name)
        return (0, int(match.group(1))) if match else (1, path.name)

    return [str(path) for path in sorted(files, key=key)]


def pick_indices(length, count):
    if length == 0:
        return []
    return np.linspace(0, length - 1, count).round().astype(int).tolist()


def readable_detection_path(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def normalized_window(boxes, width, height, margin=1.4, minimum_side=0.35):
    boxes = np.asarray(boxes, dtype=np.float64)
    left, top = boxes[:, 0].min() / width, boxes[:, 1].min() / height
    right, bottom = boxes[:, 2].max() / width, boxes[:, 3].max() / height
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    side = max((right - left) * width, (bottom - top) * height) * margin
    side = max(side, minimum_side * max(width, height))
    half_x, half_y = side / width / 2, side / height / 2
    return [
        max(center_x - half_x, 0),
        max(center_y - half_y, 0),
        min(center_x + half_x, 1),
        min(center_y + half_y, 1),
    ]


def detect_windows(records, yolo_path, device, batch_size):
    from ultralytics import YOLO

    model = YOLO(str(yolo_path))
    boxes = [[] for _ in records]
    dimensions = [None for _ in records]

    def run(modality, owners):
        paths = []
        path_owners = []
        for owner in owners:
            files = frame_files(records[owner]["modalities"].get(modality, ""))
            for frame_index in sorted(set(pick_indices(len(files), 8))):
                path = files[frame_index]
                if readable_detection_path(path):
                    paths.append(path)
                    path_owners.append(owner)
        with tqdm(total=len(paths), desc=f"YOLO {modality}") as progress:
            for start in range(0, len(paths), 512):
                chunk_paths = paths[start : start + 512]
                chunk_owners = path_owners[start : start + 512]
                results = model.predict(
                    chunk_paths,
                    classes=[0],
                    conf=0.25,
                    verbose=False,
                    device=device,
                    batch=batch_size,
                )
                for owner, result in zip(chunk_owners, results):
                    height, width = result.orig_shape
                    dimensions[owner] = (width, height)
                    if len(result.boxes):
                        best = int(result.boxes.conf.argmax())
                        boxes[owner].append(result.boxes.xyxy[best].detach().cpu().tolist())
                progress.update(len(chunk_paths))

    run("IR", range(len(records)))
    run("Depth_Color", [index for index, values in enumerate(boxes) if not values])
    return [
        normalized_window(values, *dimensions[index])
        if values and dimensions[index]
        else None
        for index, values in enumerate(boxes)
    ]


def initialize_video_worker(paths, shapes, frames, sizes):
    global VIDEO_MEMMAPS, VIDEO_CONFIG
    VIDEO_MEMMAPS = {
        size: np.memmap(path, dtype=np.uint8, mode="r+", shape=tuple(shape))
        for size, path, shape in zip(sizes, paths, shapes)
    }
    VIDEO_CONFIG = {"frames": frames, "sizes": sizes}


def read_frame(path, mode, window, size):
    try:
        with Image.open(path) as image:
            image = image.convert(mode)
            if window is not None:
                width, height = image.size
                image = image.crop(
                    (
                        round(window[0] * width),
                        round(window[1] * height),
                        round(window[2] * width),
                        round(window[3] * height),
                    )
                )
            array = np.asarray(
                image.resize((size, size), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        return None if not array.any() else array
    except (OSError, ValueError):
        return None


def process_video_record(job):
    index, record, window = job
    frames = VIDEO_CONFIG["frames"]
    buffers = {
        size: np.zeros(VIDEO_MEMMAPS[size].shape[1:], dtype=np.uint8)
        for size in VIDEO_CONFIG["sizes"]
    }
    channel = 0
    streams = [("Depth_Color", "RGB", 3), ("Thermal", "RGB", 3), ("IR", "L", 1)]
    for modality, mode, channels in streams:
        files = frame_files(record["modalities"].get(modality, ""))
        selected = pick_indices(len(files), frames)
        for output_index, source_index in enumerate(selected):
            for size in VIDEO_CONFIG["sizes"]:
                array = read_frame(files[source_index], mode, window, size)
                if array is None:
                    continue
                if channels == 1:
                    buffers[size][output_index, channel] = array
                else:
                    buffers[size][output_index, channel : channel + channels] = array.transpose(
                        2, 0, 1
                    )
        channel += channels
    for size, buffer in buffers.items():
        VIDEO_MEMMAPS[size][index] = buffer
    return index


def prepare_video(
    records,
    cache_dir,
    yolo_path,
    device,
    workers,
    detection_batch_size,
    fingerprint,
):
    windows_path = cache_dir / "windows.json"
    metadata_path = cache_dir / "runtime_manifest.json"
    sizes = [128, 160]
    frames = 32
    shapes = [(len(records), frames, 7, size, size) for size in sizes]
    paths = [cache_dir / f"video_{size}.uint8.memmap" for size in sizes]
    if metadata_path.is_file() and windows_path.is_file() and all(path.is_file() for path in paths):
        manifest = json.loads(metadata_path.read_text())
        valid = manifest.get("complete") and manifest.get("sample_fingerprint") == fingerprint
        valid &= all(path.stat().st_size == int(np.prod(shape)) for path, shape in zip(paths, shapes))
        if valid:
            return {size: path for size, path in zip(sizes, paths)}
    windows = detect_windows(records, yolo_path, device, detection_batch_size)
    windows_path.write_text(json.dumps(windows) + "\n")
    for path, shape in zip(paths, shapes):
        np.memmap(path, dtype=np.uint8, mode="w+", shape=shape).flush()
    jobs = [(record["index"], record, windows[record["index"]]) for record in records]
    with Pool(
        processes=workers,
        initializer=initialize_video_worker,
        initargs=([str(path) for path in paths], shapes, frames, sizes),
    ) as pool:
        list(tqdm(pool.imap_unordered(process_video_record, jobs, chunksize=1), total=len(jobs), desc="Video"))
    manifest = {
        "complete": True,
        "sample_fingerprint": fingerprint,
        "rows": len(records),
        "frames": frames,
        "sizes": sizes,
        "shapes": {str(size): list(shape) for size, shape in zip(sizes, shapes)},
    }
    metadata_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {size: path for size, path in zip(sizes, paths)}


def prepare_runtime(data_dir, cache_dir, yolo_path, device, workers, detection_batch_size):
    records, test_csv = discover_records(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = runtime_fingerprint(records)
    (cache_dir / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    sensors = prepare_sensors(records, cache_dir, workers, fingerprint)
    videos = prepare_video(
        records,
        cache_dir,
        yolo_path,
        device,
        workers,
        detection_batch_size,
        fingerprint,
    )
    return records, test_csv, sensors, videos


def record_interval(record, radar_only=False):
    key = (record["id"], radar_only)
    if key in INTERVAL_CACHE:
        return INTERVAL_CACHE[key]
    modalities = ("Radar",) if radar_only else ("Depth_Color", "IR", "Radar")
    for modality in modalities:
        folder = Path(record["modalities"].get(modality, ""))
        if not folder.is_dir():
            continue
        values = []
        for path in folder.iterdir():
            match = TIMESTAMP_PATTERN.search(path.name)
            if match:
                values.append(datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S.%f").timestamp())
        if values:
            INTERVAL_CACHE[key] = (min(values), max(values))
            return INTERVAL_CACHE[key]
    if radar_only:
        return record_interval(record, radar_only=False)
    imu_folder = Path(record["modalities"].get("IMU", ""))
    values = []
    if imu_folder.is_dir():
        for path in imu_folder.glob("*.csv"):
            try:
                with path.open(encoding="utf-8-sig") as handle:
                    rows = csv.reader(handle)
                    next(rows, None)
                    for row in rows:
                        if row:
                            values.append(datetime.fromisoformat(row[0]).timestamp())
            except (OSError, UnicodeDecodeError, ValueError):
                continue
    INTERVAL_CACHE[key] = (min(values), max(values)) if values else None
    return INTERVAL_CACHE[key]


def acquisition_date(record):
    interval = record_interval(record)
    if interval is None:
        return None
    return datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")


class RuntimeDataset(Dataset):
    def __init__(self, indices, records, sensors_path, video_path, image_size):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.records = records
        archive = np.load(sensors_path)
        self.skeleton = archive["skeleton"]
        self.imu = archive["imu_world"]
        self.sensor_mask = archive["sensor_mask"]
        self.image_size = image_size
        self.shape = (len(records), 32, 7, image_size, image_size)
        self.video_path = str(video_path)
        self.video_memmap = None
        mean = KINETICS_MEAN * 2 + (sum(KINETICS_MEAN) / 3,)
        std = KINETICS_STD * 2 + (sum(KINETICS_STD) / 3,)
        self.mean = torch.tensor(mean).view(1, 7, 1, 1)
        self.std = torch.tensor(std).view(1, 7, 1, 1)

    @property
    def video(self):
        if self.video_memmap is None:
            self.video_memmap = np.memmap(
                self.video_path, dtype=np.uint8, mode="r", shape=self.shape
            )
        return self.video_memmap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        video = torch.from_numpy(np.asarray(self.video[index]).copy()).float().div_(255)
        visual_mask = torch.stack(
            [
                video[:, 0:3].any().float(),
                video[:, 3:6].any().float(),
                video[:, 6:7].any().float(),
            ]
        )
        return {
            "video": video.sub_(self.mean).div_(self.std),
            "skeleton": torch.from_numpy(self.skeleton[index].copy()),
            "imu": torch.from_numpy(self.imu[index].copy()),
            "sensor_mask": torch.from_numpy(self.sensor_mask[index, :2].copy()),
            "visual_mask": visual_mask,
            "index": index,
            "label": -1,
        }


def modality_presence(sensors_path):
    archive = np.load(sensors_path)
    sensor = archive["sensor_mask"]
    video = archive["video_mask"]
    return np.stack(
        [
            video[:, 0] > 0,
            video[:, 2] > 0,
            video[:, 1] > 0,
            sensor[:, 0] > 0,
            sensor[:, 1] > 0,
            sensor[:, 2] > 0,
        ],
        axis=1,
    )
