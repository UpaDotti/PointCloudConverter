import numpy as np
import laspy
from tqdm import tqdm

from .config import Config


def write_points_bin(config: Config) -> None:
    """点群キャッシュファイルを作成"""
    center = compute_points_center(config)
    export_points_bin(config, center)


def compute_points_center(config: Config) -> np.ndarray:
    """点群の中心を計算"""

    pattern = config.options.pattern
    read_size = config.options.read_size
    las_dir = config.paths.las_dir

    # .lasファイルを取得
    las_paths = _find_las_paths(las_dir, pattern)

    # .lasが見つからないときはエラー
    if not las_paths:
        raise FileNotFoundError("LASファイルが見つかりません")

    # 最小値/ 最大値 （最初は ∞ / -∞）
    vmin = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    vmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    # .lasを開いていく / プログレスバー
    for las_path in tqdm(las_paths, desc="点群の中心を計算", unit="file"):
        with laspy.open(las_path) as reader:

            # スケール / オフセットを取得
            scale = np.array(reader.header.scales, dtype=np.float32)
            offset = np.array(reader.header.offsets, dtype=np.float32)

            # 指定の点数ごとに読む
            for pts in reader.chunk_iterator(read_size):

                # 実座標にしてUnity座標系にする
                x, y, z = _to_unity_xyz(pts, scale, offset)

                # 最小値 / 最大値 を取得
                chunk_min = np.array([x.min(), y.min(), z.min()], dtype=np.float32)
                chunk_max = np.array([x.max(), y.max(), z.max()], dtype=np.float32)

                # 最小値 / 最大値 を更新
                vmin = np.minimum(vmin, chunk_min)
                vmax = np.maximum(vmax, chunk_max)

    # 中心を計算
    center = (vmin + vmax) * 0.5

    return center


def export_points_bin(config: Config, center: np.ndarray) -> None:
    """点群のキャッシュファイルを出力"""

    read_size = config.options.read_size
    pattern = config.options.pattern
    in_dtype = config.options.in_dtype
    las_dir = config.paths.las_dir
    points_path = config.paths.points_path

    # .lasファイルを取得
    las_paths = _find_las_paths(las_dir, pattern)

    # .lasが見つからないときはエラー
    if not las_paths:
        raise FileNotFoundError("LASファイルが見つかりません")

    # NumPy配列にしてfloat32にする
    center = np.asarray(center, dtype=np.float32)

    # 出力先フォルダが無ければ作る
    points_path.parent.mkdir(parents=True, exist_ok=True)

    # ファイルが無ければ作成 / あれば中身をリセット
    with points_path.open("wb") as fp:

        # .lasを開いていく / プログレスバー
        for las_path in tqdm(las_paths, desc="キャッシュを作成", unit="file"):
            with laspy.open(las_path) as reader:

                # スケール / オフセットを取得
                scale = np.array(reader.header.scales, dtype=np.float32)
                offset = np.array(reader.header.offsets, dtype=np.float32)

                # 指定の点数ごとに読む
                for pts in reader.chunk_iterator(read_size):

                    # 実座標にしてUnity座標系にする
                    x, y, z = _to_unity_xyz(pts, scale, offset)

                    # 原点に寄せる
                    x = x.astype(np.float32, copy=False) - center[0]
                    y = y.astype(np.float32, copy=False) - center[1]
                    z = z.astype(np.float32, copy=False) - center[2]

                    # RGBを取得
                    r = pts["red"].astype(np.uint16, copy=False)
                    g = pts["green"].astype(np.uint16, copy=False)
                    b = pts["blue"].astype(np.uint16, copy=False)

                    # 指定dtypeの配列を作成
                    n = len(pts)
                    recs = np.empty(n, dtype=in_dtype)

                    # 配列に点データを詰める
                    recs["x"] = x
                    recs["y"] = y
                    recs["z"] = z
                    recs["r"] = r
                    recs["g"] = g
                    recs["b"] = b

                    # バイナリに変換して書き込み
                    recs.tofile(fp)


def _to_unity_xyz(pts, scale: np.ndarray, offset: np.ndarray):
    """実座標にしUnity座標系にする"""

    # スケールとオフセットを適応
    x = pts["X"] * scale[0] + offset[0]
    y_real = pts["Y"] * scale[1] + offset[1]
    z_real = pts["Z"] * scale[2] + offset[2]

    # Y/Z 入れ替え（Unity座標系）
    y = z_real
    z = y_real

    return x, y, z


def _find_las_paths(las_dir, pattern: str):
    """指定フォルダ配下のLASファイルを再帰的に取得する"""

    return sorted(path for path in las_dir.rglob(pattern) if path.is_file())
