from pathlib import Path
import numpy as np
from tqdm import tqdm

from .config import Config


def compute_points_aabb(config: Config) -> tuple[np.ndarray, np.ndarray, int]:
    read_size = config.options.read_size
    in_dtype = config.options.in_dtype
    points_path = config.paths.points_path

    size = points_path.stat().st_size
    total = size // in_dtype.itemsize

    vmin = np.array([np.inf, np.inf, np.inf], np.float32)
    vmax = np.array([-np.inf, -np.inf, -np.inf], np.float32)

    with points_path.open("rb") as fp, tqdm(
        total=total, desc="AABB計算", unit="pts"
    ) as pbar:
        processed = 0
        while processed < total:
            n = min(read_size, total - processed)
            recs = np.fromfile(fp, dtype=in_dtype, count=n)
            if recs.size == 0:
                break

            xyz = np.stack([recs["x"], recs["y"], recs["z"]], axis=1)
            vmin = np.minimum(vmin, xyz.min(axis=0))
            vmax = np.maximum(vmax, xyz.max(axis=0))

            processed += recs.size
            pbar.update(recs.size)

    return vmin, vmax, int(total)


def append_points(bin_path: Path, recs: np.ndarray) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with bin_path.open("ab") as f:
        recs.tofile(f)
