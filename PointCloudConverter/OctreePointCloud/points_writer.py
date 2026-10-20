from pathlib import Path

import numpy as np
import laspy
from tqdm import tqdm

from .config import Config
from .octree import NodeKey, child_index, make_child_key, partition_by_child, tmp_filename


def write_child_bins(config: Config) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """点群を8分割キャッシュへ直接振り分け、中心寄せ後の最小値 / 最大値と子ごとの点数を返す"""
    center = compute_points_center(config)
    return export_child_bins(config, center)


def compute_points_center(config: Config) -> np.ndarray:
    """点群の中心を計算"""

    pattern = config.options.pattern
    las_dir = config.paths.las_dir

    # .lasファイルを取得
    las_paths = _find_las_paths(las_dir, pattern)

    # .lasが見つからないときはエラー
    if not las_paths:
        raise FileNotFoundError("LASファイルが見つかりません")

    # 最小値/ 最大値 （最初は ∞ / -∞）
    # 実座標は値が大きく float32 では精度が足りないため float64 で扱う
    vmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    vmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    # LASヘッダーに記録された範囲を使う（全点を読まずに済む）
    for las_path in las_paths:
        with laspy.open(las_path) as reader:
            header_min = np.array(reader.header.mins, dtype=np.float64)
            header_max = np.array(reader.header.maxs, dtype=np.float64)

        # Y/Z 入れ替え（Unity座標系）
        vmin = np.minimum(vmin, header_min[[0, 2, 1]])
        vmax = np.maximum(vmax, header_max[[0, 2, 1]])

    # 中心を計算
    center = (vmin + vmax) * 0.5

    return center


def export_child_bins(config: Config, center: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """LASを読みながら8つの子キャッシュへ直接振り分け、最小値 / 最大値と子ごとの点数を返す"""

    read_size = config.options.read_size
    pattern = config.options.pattern
    in_dtype = config.options.in_dtype
    las_dir = config.paths.las_dir
    nodes_dir = config.paths.nodes_dir

    # .lasファイルを取得
    las_paths = _find_las_paths(las_dir, pattern)

    # .lasが見つからないときはエラー
    if not las_paths:
        raise FileNotFoundError("LASファイルが見つかりません")

    # NumPy配列にする（減算が終わるまで float64 のまま精度を保つ）
    center = np.asarray(center, dtype=np.float64)

    # 出力先フォルダが無ければ作る
    nodes_dir.mkdir(parents=True, exist_ok=True)

    # 8つの子キャッシュのパスを用意（前回の残骸があれば消す）
    root_key = NodeKey(depth=0, path=1)
    child_bins: dict[int, Path] = {}
    for child in range(8):
        child_key = make_child_key(root_key, child)
        child_bin = nodes_dir / tmp_filename(child_key)
        child_bin.unlink(missing_ok=True)
        child_bins[child] = child_bin

    # 中心寄せ後の最小値 / 最大値（最初は ∞ / -∞）
    # ルートノードのメタ情報として使うため、書き込みと同時に計算する
    vmin = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    vmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    # 子ごとの点数
    child_counts = np.zeros(8, dtype=np.int64)

    # 子ノードごとにファイルを開く
    fps = {child: child_bins[child].open("wb") for child in range(8)}
    try:
        # .lasを開いていく / プログレスバー
        for las_path in tqdm(las_paths, desc="キャッシュを作成", unit="file"):
            with laspy.open(las_path) as reader:

                # スケール / オフセットを取得（精度維持のため float64）
                scale = np.array(reader.header.scales, dtype=np.float64)
                offset = np.array(reader.header.offsets, dtype=np.float64)

                # 指定の点数ごとに読む
                for pts in reader.chunk_iterator(read_size):

                    # 実座標にしてUnity座標系にする
                    x, y, z = _to_unity_xyz(pts, scale, offset)

                    # 原点に寄せる（float64で引いてから、小さくなった値をfloat32に落とす）
                    x = (x - center[0]).astype(np.float32)
                    y = (y - center[1]).astype(np.float32)
                    z = (z - center[2]).astype(np.float32)

                    # 中心寄せ後の最小値 / 最大値を更新
                    chunk_min = np.array([x.min(), y.min(), z.min()], dtype=np.float32)
                    chunk_max = np.array([x.max(), y.max(), z.max()], dtype=np.float32)
                    vmin = np.minimum(vmin, chunk_min)
                    vmax = np.maximum(vmax, chunk_max)

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

                    # 中心寄せ後はルートの中心が原点になるため、分割面は0固定で8分割の行き先を決める
                    cidx = child_index(x, y, z, 0, 0, 0)

                    # 子ごとに並べ替えて連続領域を書き込み
                    sorted_recs, bounds = partition_by_child(recs, cidx)
                    for child in range(8):
                        s, e = bounds[child], bounds[child + 1]
                        if e > s:
                            sorted_recs[s:e].tofile(fps[child])
                            child_counts[child] += e - s
    finally:
        # ファイルを閉じる
        for fp in fps.values():
            fp.close()

    # 点が入らなかった子ファイルは削除
    for child in range(8):
        if child_counts[child] == 0:
            child_bins[child].unlink(missing_ok=True)

    # 点がある子だけの点数辞書を作る
    counts = {child: int(child_counts[child]) for child in range(8) if child_counts[child] > 0}

    return vmin, vmax, counts


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
