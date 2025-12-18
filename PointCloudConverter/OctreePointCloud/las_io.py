import numpy as np
import laspy
from tqdm import tqdm

from .config import Config


def compute_global_center(config: Config) -> np.ndarray:
    pattern = config.options.pattern
    las_dir = config.paths.las_dir
    las_paths = list(las_dir.glob(pattern))
    if not las_paths:
        raise FileNotFoundError("LASファイルが見つかりません")

    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0

    for las_path in tqdm(las_paths, desc="グローバル中心計算", unit="file"):
        with laspy.open(las_path) as reader:
            scale = np.array(reader.header.scales, dtype=np.float64)
            offset = np.array(reader.header.offsets, dtype=np.float64)

            for pts in reader.chunk_iterator(config.options.read_size):
                n = pts["X"].size
                if n == 0:
                    continue

                x = pts["X"].astype(np.float64) * scale[0] + offset[0]
                y_real = pts["Y"].astype(np.float64) * scale[1] + offset[1]
                z_real = pts["Z"].astype(np.float64) * scale[2] + offset[2]

                # Unity座標系：Y/Z 入れ替え
                y = z_real
                z = y_real

                sum_x += x.sum()
                sum_y += y.sum()
                sum_z += z.sum()
                count += n

    if count == 0:
        raise ValueError("点が0件でした（LASに点が入っていない可能性）")

    return np.array([sum_x / count, sum_y / count, sum_z / count], dtype=np.float32)


def export_las_to_points_bin(config: Config, global_center: np.ndarray) -> None:
    read_size = config.options.read_size
    pattern = config.options.pattern
    in_dtype = config.options.in_dtype
    las_dir = config.paths.las_dir
    points_path = config.paths.points_path

    las_paths = list(las_dir.glob(pattern))
    if not las_paths:
        raise FileNotFoundError("LASファイルが見つかりません")

    global_center = np.asarray(global_center, dtype=np.float32)
    points_path.parent.mkdir(parents=True, exist_ok=True)

    with points_path.open("wb") as fp:
        for las_path in tqdm(las_paths, desc="LAS→points.bin Export", unit="file"):
            with laspy.open(las_path) as reader:
                scale = np.array(reader.header.scales, dtype=np.float32)
                offset = np.array(reader.header.offsets, dtype=np.float32)

                for pts in reader.chunk_iterator(read_size):
                    n = pts["X"].size
                    if n == 0:
                        continue

                    x = pts["X"].astype(np.float32, copy=False) * scale[0] + offset[0]
                    y_real = (
                        pts["Y"].astype(np.float32, copy=False) * scale[1] + offset[1]
                    )
                    z_real = (
                        pts["Z"].astype(np.float32, copy=False) * scale[2] + offset[2]
                    )

                    y = z_real
                    z = y_real

                    x -= global_center[0]
                    y -= global_center[1]
                    z -= global_center[2]

                    # RGB16（無いLASだと落ちるので必要ならガードを入れてね）
                    r = pts["red"].astype(np.uint16, copy=False)
                    g = pts["green"].astype(np.uint16, copy=False)
                    b = pts["blue"].astype(np.uint16, copy=False)

                    recs = np.empty(n, dtype=in_dtype)
                    recs["x"] = x
                    recs["y"] = y
                    recs["z"] = z
                    recs["r"] = r
                    recs["g"] = g
                    recs["b"] = b

                    recs.tofile(fp)
