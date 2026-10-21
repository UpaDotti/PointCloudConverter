import collections
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from PointCloudConverter import utils
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


class PointStore:
    """点群を1つのファイルに追記し、追記位置（点数単位のオフセット）を返す"""

    def __init__(self, path: Path):
        self.fp = path.open("wb")
        self.count = 0

    def append(self, pts: np.ndarray) -> int:
        offset = self.count
        pts.tofile(self.fp)
        self.count += int(pts.shape[0])
        return offset

    def close(self) -> None:
        self.fp.close()


class MemoryStore:
    """PointStoreと同じ使い方でメモリに溜める（並列処理の結果をメインプロセスに返すため）"""

    def __init__(self, dtype: np.dtype):
        self.dtype = dtype
        self.parts: list[np.ndarray] = []
        self.count = 0

    def append(self, pts: np.ndarray) -> int:
        offset = self.count
        self.parts.append(pts)
        self.count += int(pts.shape[0])
        return offset

    def data(self) -> np.ndarray:
        if not self.parts:
            return np.empty(0, dtype=self.dtype)
        return np.concatenate(self.parts)


@dataclass
class BuildState:
    """オクトツリー構築中の出力先"""

    branches: list[dict]
    leaves: list[dict]
    leaf_store: PointStore | MemoryStore
    branch_store: PointStore | MemoryStore
    node_count: int = 0  # 処理したノード数（プログレスバー用）


def make_child_key(parent: NodeKey, child: int) -> NodeKey:
    depth = parent.depth + 1
    path = (parent.path << 3) | child
    return NodeKey(depth=depth, path=path)


# 葉 / 枝の代表点はノードごとにファイルを分けず、1つのファイルに追記してオフセットで参照する
# （Windowsでは新規ファイルを初めて開くたびにウイルススキャンの待ちが入り、ノード数が多いと大きな時間になるため）
LEAVES_FILENAME = "leaves.bin"  # 葉の点（量子化済み / out_dtype）
BRANCHES_FILENAME = "branches.bin"  # 枝の代表点と、代表点生成用に間引いた葉の点（in_dtype）


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
    max_points_in_memory = config.options.max_points_in_memory
    in_dtype = config.options.in_dtype
    nodes_dir = config.paths.nodes_dir

    # 出力先フォルダが無ければ作る
    nodes_dir.mkdir(parents=True, exist_ok=True)

    # 分解キュー（呼び出し元から振り分け済みノードを受け取る）
    queue = dict(queue)
    branches = list(branches)
    leaves = []

    # 葉 / 枝の代表点の書き出し先
    leaf_store = PointStore(nodes_dir / LEAVES_FILENAME)
    branch_store = PointStore(nodes_dir / BRANCHES_FILENAME)

    # 並列処理中の部分木（結果がメモリに溜まりすぎないよう同時実行数を制限）
    pending: collections.deque = collections.deque()

    def merge(result) -> None:
        """部分木の結果を葉 / 枝の代表点ファイルに書き込み、オフセットをファイル内の位置に直す"""
        sub_branches, sub_leaves, leaf_data, branch_data, node_count = result
        leaf_base = leaf_store.append(leaf_data)
        branch_base = branch_store.append(branch_data)
        for leaf in sub_leaves:
            leaf["offset"] += leaf_base
            if "sample" in leaf:
                leaf["sample"]["offset"] += branch_base
        for branch in sub_branches:
            branch["lod"]["offset"] += branch_base
        branches.extend(sub_branches)
        leaves.extend(sub_leaves)
        pbar.update(node_count)

    # プログレスバー / 並列処理用のプロセスプール
    with tqdm(desc="オクトツリー分割", unit="node") as pbar, utils.process_pool(
        config.options.worker_count()
    ) as pool:
        window = pool._processes * 2 if pool else 0

        # キューがある間はループ
        while queue or pending:

            # 終わった部分木の結果を取り込む（キューが空なら終わるまで待つ）
            while pending and (not queue or pending[0].ready() or len(pending) >= window):
                merge(pending.popleft().get())
            if not queue:
                continue

            # キューを一個取り出す
            key, work = queue.popitem()
            bin_path = work.bin_path
            vmin = work.vmin
            vmax = work.vmax
            count = work.count

            # 分割するかの判定（深さが上限未満 + 点数が指定を超える）
            should_split = key.depth < max_depth and count > max_points_per_leaf

            # メモリに載る点数なら部分木をまとめてメモリ上で分割する（部分木ごとに並列処理）
            if count <= max_points_in_memory:
                task = (config, key, bin_path, vmin, vmax)
                if pool:
                    pending.append(pool.apply_async(build_subtree_task, (task,)))
                else:
                    merge(build_subtree_task(task))

            # 分割しない場合は葉ノードにする（メモリに載らない葉はチャンクごとに処理）
            elif not should_split:
                tight_min, tight_max = file_bounds(config, bin_path)
                mm = np.memmap(bin_path, mode="r", dtype=in_dtype)
                sample = write_leaf_sample(config, mm, branch_store)
                offset = quantize_file(config, mm, leaf_store, tight_min, tight_max)
                mm._mmap.close()
                del mm
                bin_path.unlink(missing_ok=True)
                record = leaf_record(key, vmin, vmax, count, offset, tight_min, tight_max)
                record["sample"] = sample
                leaves.append(record)
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
                branches.append(branch_record(key, vmin, vmax, count, child_flags))

                # プログレスバーを更新
                pbar.update(1)

            pbar.set_postfix(
                depth=key.depth,
                leaves=len(leaves),
                queued=len(queue),
            )

    leaf_store.close()
    branch_store.close()

    branches_sorted = sorted(
        branches, key=lambda branch: (branch["key"]["depth"], branch["key"]["path"])
    )

    leaves_sorted = sorted(
        leaves, key=lambda leaf: (leaf["key"]["depth"], leaf["key"]["path"])
    )

    return branches_sorted, leaves_sorted


def build_subtree_task(task):
    """
    1つの部分木をメモリ上で分割する（並列処理用）
    葉 / 枝の代表点は書き出さずに返し、メインプロセスがまとめて書き込む（オフセットは部分木内の位置）
    """

    config, key, bin_path, vmin, vmax = task
    max_depth = config.options.max_depth
    max_points_per_leaf = config.options.max_points_per_leaf

    # 一度だけ読み込む
    pts = np.fromfile(bin_path, dtype=config.options.in_dtype)
    bin_path.unlink(missing_ok=True)
    count = int(pts.shape[0])

    # 部分木を分割
    leaf_store = MemoryStore(config.options.out_dtype)
    branch_store = MemoryStore(config.options.in_dtype)
    state = BuildState([], [], leaf_store, branch_store)
    summary = build_subtree_in_memory(config, key, pts, vmin, vmax, state)

    # 部分木の根が葉なら、親（ファイル上で分割した枝）の代表点用に間引いた点を残す
    if not (key.depth < max_depth and count > max_points_per_leaf):
        state.leaves[-1]["sample"] = write_leaf_sample(config, summary, branch_store)

    return state.branches, state.leaves, leaf_store.data(), branch_store.data(), state.node_count


def build_subtree_in_memory(
    config: Config,
    key: NodeKey,
    pts: np.ndarray,
    vmin,
    vmax,
    state: BuildState,
) -> np.ndarray:
    """
    メモリ上の点群を再帰的に8分割し、葉（量子化済み）と枝の代表点をファイルに書き出す
    戻り値は親が代表点を間引く元になる点群（葉なら生点 / 枝なら代表点）
    """

    max_depth = config.options.max_depth
    max_points_per_leaf = config.options.max_points_per_leaf
    out_dtype = config.options.out_dtype
    count = int(pts.shape[0])

    # 分割しない場合は葉ノードにする（保存用の実範囲で量子化して書き出す）
    should_split = key.depth < max_depth and count > max_points_per_leaf
    if not should_split:
        tight_min, tight_max = array_bounds(pts)
        offset = state.leaf_store.append(utils.quantize_points(pts, tight_min, tight_max, out_dtype))
        state.leaves.append(leaf_record(key, vmin, vmax, count, offset, tight_min, tight_max))
        state.node_count += 1
        return pts

    # 子ごとに連続するよう並べ替える
    center = (vmin + vmax) * 0.5
    center_x, center_y, center_z = center.tolist()
    cidx = child_index(pts["x"], pts["y"], pts["z"], center_x, center_y, center_z)
    sorted_pts, bounds = partition_by_child(pts, cidx)
    del cidx

    # 枝メタファイルを追加
    child_flags = [bool(bounds[child + 1] > bounds[child]) for child in range(8)]
    record = branch_record(key, vmin, vmax, count, child_flags)
    state.branches.append(record)
    state.node_count += 1

    # 点がある子だけ再帰し、子から代表点を集める（ボトムアップ）
    max_per_child = max(1, config.options.max_points_per_branch // 8)
    chunks = []
    for child in range(8):
        if not child_flags[child]:
            continue
        mn, mx = child_aabb(vmin, vmax, child)
        part = sorted_pts[bounds[child] : bounds[child + 1]]
        child_summary = build_subtree_in_memory(
            config, make_child_key(key, child), part, mn, mx, state
        )
        chunks.append(sample_for_parent(child_summary, max_per_child))
    del sorted_pts

    # 代表点を書き出し、メタを枝に記録
    summary = np.concatenate(chunks)
    offset = state.branch_store.append(summary)
    tight_min, tight_max = array_bounds(summary)
    record["lod"] = {
        "point_count": int(summary.size),
        "offset": offset,
        "min": tight_min,
        "max": tight_max,
    }

    return summary


def sample_for_parent(src: np.ndarray, max_per_child: int) -> np.ndarray:
    """親の代表点用に、固定間隔で間引いて上限まで取る"""

    n = int(src.shape[0])
    step = max(1, n // max_per_child)
    idx = utils.downsample_indices(n, step=step)
    if idx.size > max_per_child:
        idx = idx[:max_per_child]
    return np.array(src[idx])


def write_leaf_sample(config: Config, src: np.ndarray, store: PointStore) -> dict:
    """親の代表点用に間引いた葉の点を書き出し、位置と点数を返す"""

    max_per_child = max(1, config.options.max_points_per_branch // 8)
    sample = sample_for_parent(src, max_per_child)
    offset = store.append(sample)
    return {"offset": offset, "point_count": int(sample.size)}


def quantize_file(config: Config, mm: np.ndarray, store: PointStore, vmin, vmax) -> int:
    """memmap上の点群をチャンクごとに量子化して書き出し、先頭の位置を返す"""

    read_size = config.options.read_size
    out_dtype = config.options.out_dtype
    offset = store.count
    for start in range(0, mm.shape[0], read_size):
        part = np.array(mm[start : start + read_size])
        store.append(utils.quantize_points(part, vmin, vmax, out_dtype))
    return offset


def leaf_record(key: NodeKey, vmin, vmax, count: int, offset: int, tight_min, tight_max) -> dict:
    """葉のメタデータ（min/maxは幾何セル、tight_min/tight_maxは実際の点の範囲、offsetは葉ファイル内の位置）"""

    return {
        "key": key_to_dict(key),
        "min": np.asarray(vmin, np.float32).tolist(),
        "max": np.asarray(vmax, np.float32).tolist(),
        "count": int(count),
        "offset": int(offset),
        "tight_min": tight_min,
        "tight_max": tight_max,
    }


def branch_record(key: NodeKey, vmin, vmax, count: int, child_flags: list[bool]) -> dict:
    """枝のメタデータ"""

    return {
        "key": key_to_dict(key),
        "min": np.asarray(vmin, np.float32).tolist(),
        "max": np.asarray(vmax, np.float32).tolist(),
        "count": int(count),
        "children": child_flags,
    }


def array_bounds(pts: np.ndarray) -> tuple[list[float], list[float]]:
    """点群の実範囲(tight)"""

    bounds_min = [float(pts["x"].min()), float(pts["y"].min()), float(pts["z"].min())]
    bounds_max = [float(pts["x"].max()), float(pts["y"].max()), float(pts["z"].max())]
    return bounds_min, bounds_max


def file_bounds(config: Config, node_bin: Path) -> tuple[list[float], list[float]]:
    """binファイル内の点群の実範囲(tight)をチャンクごとに読んで計算"""

    read_size = config.options.read_size
    mm = np.memmap(node_bin, mode="r", dtype=config.options.in_dtype)
    vmin = np.full(3, np.inf, dtype=np.float32)
    vmax = np.full(3, -np.inf, dtype=np.float32)
    for start in range(0, mm.shape[0], read_size):
        chunk_min, chunk_max = array_bounds(np.array(mm[start : start + read_size]))
        vmin = np.minimum(vmin, chunk_min)
        vmax = np.maximum(vmax, chunk_max)
    mm._mmap.close()
    del mm
    return [float(v) for v in vmin], [float(v) for v in vmax]


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

        # 読む箇所を切り出し（連続領域としてRAMに読み込む）
        end = start + read_size
        part = np.array(mm[start:end])

        # 8分割のどの位置にいるか
        cidx = child_index(part["x"], part["y"], part["z"], center_x, center_y, center_z)

        # 子ごとに並べ替えて連続領域を書き込み 点数を更新
        sorted_part, bounds = partition_by_child(part, cidx)
        for child in range(8):
            s, e = bounds[child], bounds[child + 1]
            if e > s:
                sorted_part[s:e].tofile(fps[child])
                child_counts[child] += e - s

        # 参照を解除
        del part, sorted_part

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


def child_index(x, y, z, center_x, center_y, center_z) -> np.ndarray:
    """8分割のどの位置にいるかを3bitで表現"""

    return (
        ((x >= center_x).astype(np.uint8) << 2)
        | ((y >= center_y).astype(np.uint8) << 1)
        | ((z >= center_z).astype(np.uint8) << 0)
    )


def partition_by_child(recs: np.ndarray, cidx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """子ごとに連続するよう並べ替え、並べ替え後の配列と子ごとの区切り位置を返す（子内の順序は維持）"""

    # 安定ソートで子番号順に並べる（uint8なので基数ソートになり高速）
    order = np.argsort(cidx, kind="stable")

    # 構造体のまま並べ替えるより、同サイズのバイト列(void)として並べ替えるほうが速い
    void_dtype = np.dtype((np.void, recs.dtype.itemsize))
    sorted_recs = recs.view(void_dtype)[order].view(recs.dtype)

    # 子ごとの点数から区切り位置を作る
    counts = np.bincount(cidx, minlength=8)
    bounds = np.zeros(9, dtype=np.int64)
    np.cumsum(counts, out=bounds[1:])

    return sorted_recs, bounds


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
