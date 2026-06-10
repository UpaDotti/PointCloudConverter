from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from .config import Config


# ------------------------
# データクラス
# ------------------------
@dataclass(frozen=True)
class NodeWork:
    bin_path: Path
    vmin: np.ndarray
    vmax: np.ndarray
    count: int


@dataclass(frozen=True)
class NodeKey:
    depth: int
    path: int


def make_child_key(parent: NodeKey, child: int) -> NodeKey:
    depth = parent.depth + 1
    path = (parent.path << 3) | child
    return NodeKey(depth=depth, path=path)


def leaf_filename(key: NodeKey) -> str:
    return f"leaf_d{key.depth}_p{key.path}.bin"


def branch_filename(key: NodeKey) -> str:
    return f"branch_d{key.depth}_p{key.path}.bin"


def tmp_filename(key: NodeKey) -> str:
    return f"tmp_d{key.depth}_p{key.path}.bin"


def key_to_dict(k: NodeKey) -> dict:
    return {"depth": int(k.depth), "path": int(k.path)}


def dict_to_key(d: dict) -> NodeKey:
    return NodeKey(depth=int(d["depth"]), path=int(d["path"]))


# ------------------------
# オクトツリー構築
# ------------------------
def build_octree(
    config: Config,
    points_min: np.ndarray,
    points_max: np.ndarray,
    child_counts: dict[int, int],
) -> None:
    """オクトツリーを構築（ルートはキャッシュ作成時に8分割済み）"""

    max_depth = config.options.max_depth
    max_points_per_leaf = config.options.max_points_per_leaf
    nodes_dir = config.paths.nodes_dir
    oct_manifest_path = config.paths.oct_manifest_path

    # ルートの範囲はキャッシュ作成時に計算済みの最小値 / 最大値を使う
    root_min = np.asarray(points_min, np.float32)
    root_max = np.asarray(points_max, np.float32)

    # 合計点数を計算
    total = sum(child_counts.values())

    # ルートキー
    root_key = NodeKey(depth=0, path=1)

    # 振り分け済みの子キャッシュをキューに積む
    queue: dict[NodeKey, NodeWork] = {}
    child_flags = [False] * 8
    for child, count in child_counts.items():
        child_flags[child] = True
        child_key = make_child_key(root_key, child)
        child_bin = nodes_dir / tmp_filename(child_key)
        child_min, child_max = child_aabb(root_min, root_max, child)
        queue[child_key] = NodeWork(child_bin, child_min, child_max, count)

    # ルートは振り分け済みなので最初から枝として記録
    root_branch = {
        "key": key_to_dict(root_key),
        "min": root_min.tolist(),
        "max": root_max.tolist(),
        "count": total,
        "children": child_flags,
    }

    # 子キャッシュを構築
    branches, leaves = build_octree_bins(config, queue, [root_branch])

    # 全体のメタファイルを作成
    manifest = {
        "root_min": np.asarray(root_min, np.float32).tolist(),
        "root_max": np.asarray(root_max, np.float32).tolist(),
        "total_points": total,
        "max_depth": max_depth,
        "max_points_per_leaf": max_points_per_leaf,
        "branches": branches,
        "leaves": leaves,
    }

    # メタファイルを出力
    with oct_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_octree_bins(
    config: Config,
    queue: dict[NodeKey, NodeWork],
    branches: list[dict],
) -> tuple[list[dict], list[dict]]:
    """オクトツリーのキャッシュを構築"""

    max_depth = config.options.max_depth
    max_points_per_leaf = config.options.max_points_per_leaf
    nodes_dir = config.paths.nodes_dir

    # 出力先フォルダが無ければ作る
    nodes_dir.mkdir(parents=True, exist_ok=True)

    # 分解キュー（呼び出し元から振り分け済みノードを受け取る）
    queue = dict(queue)
    branches = list(branches)
    leaves = []

    # プログレスバー
    with tqdm(desc="オクトツリー分割", unit="node") as pbar:

        # キューがある間はループ
        while queue:

            # キューを一個取り出す
            key, work = queue.popitem()
            bin_path = work.bin_path
            vmin = work.vmin
            vmax = work.vmax
            count = work.count

            # 分割するかの判定（深さが上限未満 + 点数が指定を超える）
            should_split = key.depth < max_depth and count > max_points_per_leaf

            # 分割しない場合は葉ノードにする
            if not should_split:
                leaf_bin = nodes_dir / leaf_filename(key)
                leaf_bin.unlink(missing_ok=True)
                bin_path.replace(leaf_bin)
                leaves.append(
                    {
                        "key": key_to_dict(key),
                        "min": vmin.tolist(),
                        "max": vmax.tolist(),
                        "count": count,
                        "file": leaf_bin.name,
                    }
                )
                pbar.update(1)
            else:
                # 子ノードに分割
                split = split_node_file(config, bin_path, key, vmin, vmax)

                # 子ノードをキューに追加
                child_flags = [False] * 8
                for child, (child_bin, mn, mx, cn) in split.items():
                    child_flags[child] = True
                    child_key = make_child_key(key, child)
                    queue[child_key] = NodeWork(child_bin, mn, mx, int(cn))

                # 枝メタファイルを追加
                branches.append(
                    {
                        "key": key_to_dict(key),
                        "min": np.asarray(vmin, np.float32).tolist(),
                        "max": np.asarray(vmax, np.float32).tolist(),
                        "count": count,
                        "children": child_flags,
                    }
                )

                # プログレスバーを更新
                pbar.update(1)

            pbar.set_postfix(
                depth=key.depth,
                leaves=len(leaves),
                queued=len(queue),
            )

    branches_sorted = sorted(
        branches, key=lambda branch: (branch["key"]["depth"], branch["key"]["path"])
    )

    leaves_sorted = sorted(
        leaves, key=lambda leaf: (leaf["key"]["depth"], leaf["key"]["path"])
    )

    return branches_sorted, leaves_sorted


def split_node_file(
    config: Config,
    node_bin: Path,
    parent_key: NodeKey,
    vmin,
    vmax,
) -> dict[int, tuple[Path, np.ndarray, np.ndarray, int]]:
    """8つのノードに分割する関数"""

    read_size = config.options.read_size
    in_dtype = config.options.in_dtype
    nodes_dir = config.paths.nodes_dir

    # 中心を計算
    center = (vmin + vmax) * 0.5
    center_x, center_y, center_z = center.tolist()

    # memmap : すべてをRAMに乗せるのを回避するため
    mm = np.memmap(node_bin, mode="r", dtype=in_dtype)

    # 親ノードに点がない場合は削除
    n = mm.shape[0]
    if n == 0:
        mm._mmap.close()
        del mm
        node_bin.unlink(missing_ok=True)
        return {}

    # 一時ファイル
    child_bins: dict[int, Path] = {}
    for child in range(8):
        child_key = make_child_key(parent_key, child)
        child_bin = nodes_dir / tmp_filename(child_key)
        child_bin.unlink(missing_ok=True)
        child_bins[child] = child_bin

    # 子ノードごとにファイルを開く
    fps: dict[int, object] = {}
    for child in range(8):
        fps[child] = child_bins[child].open("ab")

    #  点群を書き込み
    child_counts = np.zeros(8, dtype=np.int64)
    for start in range(0, n, read_size):

        # 読む箇所を切り出し
        end = start + read_size
        part = mm[start:end]

        # 8分割のどの位置にいるかを3bitで表現
        part_x, part_y, part_z = part["x"], part["y"], part["z"]
        cidx = (
            ((part_x >= center_x).astype(np.uint8) << 2)
            | ((part_y >= center_y).astype(np.uint8) << 1)
            | ((part_z >= center_z).astype(np.uint8) << 0)
        )

        # 書き込み 点数を更新
        for child in range(8):
            sel = cidx == child
            if np.any(sel):
                recs = part[sel]
                recs.tofile(fps[child])
                child_counts[child] += recs.size

        # 参照を解除
        del part

    # ファイルを閉じる
    for fp in fps.values():
        fp.close()

    # memmapを閉じる
    mm._mmap.close()
    del mm

    # 元のモードを削除
    node_bin.unlink(missing_ok=True)

    # 子ノードの情報を構築
    out: dict[int, tuple[Path, np.ndarray, np.ndarray, int]] = {}
    for child in range(8):
        count = child_counts[child]
        if count <= 0:
            child_bins[child].unlink(missing_ok=True)
            continue
        mn, mx = child_aabb(vmin, vmax, child)
        out[child] = (child_bins[child], mn, mx, count)

    return out


def child_aabb(vmin, vmax, cidx: int):
    """分割後の最小最大を計算"""

    vmin = np.asarray(vmin, np.float32)
    vmax = np.asarray(vmax, np.float32)
    center = (vmin + vmax) * 0.5

    x = (cidx >> 2) & 1
    y = (cidx >> 1) & 1
    z = (cidx >> 0) & 1
    is_upper = np.array([x, y, z], dtype=np.uint8)

    child_min = np.where(is_upper == 0, vmin, center)
    child_max = np.where(is_upper == 0, center, vmax)

    return child_min, child_max
