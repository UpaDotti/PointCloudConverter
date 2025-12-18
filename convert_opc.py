# 標準ライブラリ
from pathlib import Path
import json
from dataclasses import dataclass
import struct
import shutil
from dataclasses import dataclass, field
from typing import Tuple, TypedDict

# サードパーティー
import laspy
import numpy as np
from tqdm import tqdm

# 自作モジュール
import utils


# ------------------------
# 設定
# ------------------------
@dataclass(frozen=True)
class Options:
    # magic version
    magic = b"TPC\x00"
    version_major = 1
    version_minor = 0
    version_patch = 0

    # octree
    max_depth: int = 8
    max_points_per_leaf: int = 200_000

    # lod
    lod_levels: int = 7
    lod_first_step: int = 1
    lod_step_factor: int = 4
    lod_steps: Tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        steps = utils.make_lod_steps(
            self.lod_levels, self.lod_first_step, self.lod_step_factor
        )
        object.__setattr__(self, "lod_steps", tuple(map(int, steps)))

    read_size: int = 10_000

    pattern: str = "*.las"

    # PointDType
    point_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("r", "<u2"),
            ("g", "<u2"),
            ("b", "<u2"),
        ]
    )


@dataclass(frozen=True)
class Paths:
    las_dir: Path
    out_dir: Path

    @property
    def opc_path(self) -> Path:
        return self.out_dir / "sample.opc"

    @property
    def points_path(self) -> Path:
        return self.out_dir / "points.bin"

    @property
    def nodes_dir(self) -> Path:
        return self.out_dir / "nodes"

    @property
    def oct_manifest_path(self) -> Path:
        return self.out_dir / "octree_manifest.json"

    @property
    def leaf_lods_path(self) -> Path:
        return self.out_dir / "leaf_lods.json"


@dataclass(frozen=True)
class Config:
    options: Options
    paths: Paths


# ------------------------
# 1) 中心を算出
# ------------------------
def compute_global_center(config: Config) -> np.ndarray:
    """
    指定ディレクトリ内のLASファイル群からグローバル中心を計算する。
    """

    # 参照を取得
    pattern = config.options.pattern
    las_dir = config.paths.las_dir

    # 対象ファイル一覧を集める
    las_paths = list(las_dir.glob(pattern))

    # ファイルが1つも無い場合は止める
    assert len(las_paths) > 0, "LASファイルが見つかりません"

    # 計算用の変数
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0

    # 各LASファイルを順番に読む
    for las_path in tqdm(las_paths, desc="グローバル中心計算", unit="file"):
        with laspy.open(las_path) as reader:
            scale = np.array(reader.header.scales, dtype=np.float64)
            offset = np.array(reader.header.offsets, dtype=np.float64)

            for pts in reader.chunk_iterator(config.options.read_size):
                n = pts["X"].size
                if n == 0:
                    continue

                # 実座標
                x = pts["X"].astype(np.float64) * scale[0] + offset[0]
                y_real = pts["Y"].astype(np.float64) * scale[1] + offset[1]
                z_real = pts["Z"].astype(np.float64) * scale[2] + offset[2]

                # Unity座標系：Y/Z 入れ替え
                y = z_real
                z = y_real

                # 変数に追加
                sum_x += x.sum()
                sum_y += y.sum()
                sum_z += z.sum()
                count += n

    # 中心を算出
    center = np.array(
        [sum_x / count, sum_y / count, sum_z / count],
        dtype=np.float32,
    )

    return center


# ------------------------
# 2) 点群をまとめたキャッシュファイルを出力
# ------------------------
def export_las_to_points_bin(config: Config, global_center: np.ndarray) -> None:
    """
    LASファイル群を読み込み、points.bin に連続で書き出す。
    """

    # 参照を取得
    read_size = config.options.read_size
    pattern = config.options.pattern
    point_dtype = config.options.point_dtype
    las_dir = config.paths.las_dir
    points_path = config.paths.points_path

    # 対象ファイル一覧を集める
    las_paths = list(las_dir.glob(pattern))

    # ファイルが1つも無い場合は止める
    assert len(las_paths) > 0, "LASファイルが見つかりません"

    # float32に変更
    global_center = np.asarray(global_center, dtype=np.float32)

    # 出力先のフォルダを作成
    points_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with points_path.open("wb") as fp:
        for las_path in tqdm(las_paths, desc="LAS→points.bin Export", unit="file"):
            with laspy.open(las_path) as reader:
                scale = np.array(reader.header.scales, dtype=np.float32)
                offset = np.array(reader.header.offsets, dtype=np.float32)

                for pts in reader.chunk_iterator(read_size):
                    n = pts["X"].size
                    if n == 0:
                        continue

                    # 実座標化
                    x = pts["X"].astype(np.float32, copy=False) * scale[0] + offset[0]
                    y_real = (
                        pts["Y"].astype(np.float32, copy=False) * scale[1] + offset[1]
                    )
                    z_real = (
                        pts["Z"].astype(np.float32, copy=False) * scale[2] + offset[2]
                    )

                    # Y/Zを入替
                    y = z_real
                    z = y_real

                    # センタリング
                    x -= global_center[0]
                    y -= global_center[1]
                    z -= global_center[2]

                    # RGB16
                    r = pts["red"].astype(np.uint16, copy=False)
                    g = pts["green"].astype(np.uint16, copy=False)
                    b = pts["blue"].astype(np.uint16, copy=False)

                    # まとめて書き出し
                    recs = np.empty(n, dtype=point_dtype)
                    recs["x"] = x
                    recs["y"] = y
                    recs["z"] = z
                    recs["r"] = r
                    recs["g"] = g
                    recs["b"] = b

                    recs.tofile(fp)
                    total += n


# ------------------------
# 3) 点オクトツリーを構築
# ------------------------
def build_octree(config: Config) -> None:

    # 参照を取得
    read_size = config.options.read_size
    point_dtype = config.options.point_dtype
    max_depth = config.options.max_depth
    max_points_per_leaf = config.options.max_points_per_leaf
    nodes_dir = config.paths.nodes_dir
    points_path = config.paths.points_path
    oct_manifest_path = config.paths.oct_manifest_path

    # 出力フォルダを作成
    nodes_dir.mkdir(parents=True, exist_ok=True)

    # 全体AABBと総点数を計算
    root_min, root_max, total = compute_points_aabb(config)

    # ルートノード
    root_key = NodeKey(depth=0, path=1)
    root_bin = nodes_dir / "root.bin"

    if root_bin.exists():
        root_bin.unlink()

    # 全点を root.bin に書き込む
    with points_path.open("rb") as fp, tqdm(
        total=total, desc="root作成", unit="pts"
    ) as pbar:
        processed = 0
        while processed < total:
            n = min(read_size, total - processed)
            recs = np.fromfile(fp, dtype=point_dtype, count=n)
            if recs.size == 0:
                break
            append_points(root_bin, recs)
            processed += int(recs.size)
            pbar.update(int(recs.size))

    # 処理待ちノード
    queue = {root_key: (root_bin, root_min, root_max, total)}

    # 分割結果
    branches = []
    leaves = []

    # オクトツリー分割
    with tqdm(desc="オクトツリー分割", unit="node") as pbar:
        while queue:
            key, (bin_path, vmin, vmax, count) = queue.popitem()
            pbar.update(1)

            # 分割するか判定
            should_split = key.depth < max_depth and count > max_points_per_leaf

            # 分割しない → 葉ノード
            if not should_split:
                leaf_bin = nodes_dir / leaf_filename(key)
                if leaf_bin.exists():
                    leaf_bin.unlink()
                bin_path.replace(leaf_bin)
                leaves.append(
                    {
                        "key": key_to_dict(key),
                        "min": np.asarray(vmin, np.float32).tolist(),
                        "max": np.asarray(vmax, np.float32).tolist(),
                        "count": int(count),
                        "file": leaf_bin.name,
                    }
                )
                continue

            # 分割する → 8分割（ストリーミング版）
            split = split_node_file(config, bin_path, key, vmin, vmax)
            child_flags = [False] * 8

            # 子ノードを queue に追加（もう child_bin は出来てる）
            for cid, (child_bin, mn, mx, cn) in split.items():
                child_flags[cid] = True
                child_key = NodeKey(depth=key.depth + 1, path=(key.path << 3) | cid)
                queue[child_key] = (child_bin, mn, mx, cn)

            # 内部ノードとして記録
            branches.append(
                {
                    "key": key_to_dict(key),
                    "min": np.asarray(vmin, np.float32).tolist(),
                    "max": np.asarray(vmax, np.float32).tolist(),
                    "count": int(count),
                    "children": child_flags,
                }
            )

    # manifestを書き出す
    manifest = {
        "root_min": np.asarray(root_min, np.float32).tolist(),
        "root_max": np.asarray(root_max, np.float32).tolist(),
        "max_depth": int(max_depth),
        "max_points_per_leaf": int(max_points_per_leaf),
        "branches": branches,
        "leaves": leaves,
    }

    # JSONに保存
    with oct_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def compute_points_aabb(config: Config) -> tuple[np.ndarray, np.ndarray, int]:

    # 参照を取得
    read_size = config.options.read_size
    point_dtype = config.options.point_dtype
    points_path = config.paths.points_path

    # 総点数を計算
    size = points_path.stat().st_size
    total = size // point_dtype.itemsize

    # 記録用の変数を初期化
    vmin = np.array([np.inf, np.inf, np.inf], np.float32)
    vmax = np.array([-np.inf, -np.inf, -np.inf], np.float32)

    with points_path.open("rb") as fp, tqdm(
        total=total, desc="AABB計算", unit="pts"
    ) as pbar:

        processed = 0
        while processed < total:

            # 一度に読み込む点数を制限
            n = min(read_size, total - processed)

            # 指定個数分の点を読み込む
            recs = np.fromfile(fp, dtype=point_dtype, count=n)

            # ない場合はループ終了
            if recs.size == 0:
                break

            # xyzを配列にする
            xyz = np.stack([recs["x"], recs["y"], recs["z"]], axis=1)

            # 最小最大を求める
            vmin = np.minimum(vmin, xyz.min(axis=0))
            vmax = np.maximum(vmax, xyz.max(axis=0))

            # 処理済み点数を更新
            processed += recs.size

            # 進行バーを更新
            pbar.update(recs.size)

    return vmin, vmax, int(total)


def append_points(bin_path: Path, recs: np.ndarray) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with bin_path.open("ab") as f:
        recs.tofile(f)


def split_node_file(
    config: Config,
    node_bin: Path,
    parent_key: "NodeKey",
    vmin,
    vmax,
) -> dict[int, tuple[Path, np.ndarray, np.ndarray, int]]:
    """
    node_bin を読み、点を 8 子ノードのファイルへ「追記」しながら分割する。
    メモリに溜めないので大規模データでも安全。

    戻り値:
      { cid: (child_bin_path, child_min, child_max, child_count) }
    """

    # 参照を取得
    read_size = config.options.read_size
    point_dtype = config.options.point_dtype
    nodes_dir = config.paths.nodes_dir

    # 親AABB中心
    vmin = np.asarray(vmin, np.float32)
    vmax = np.asarray(vmax, np.float32)
    center = (vmin + vmax) * 0.5
    cx, cy, cz = center.tolist()

    mm = np.memmap(node_bin, mode="r", dtype=point_dtype)
    n = int(mm.shape[0])
    if n == 0:
        mm._mmap.close()
        del mm
        node_bin.unlink(missing_ok=True)
        return {}

    # 子ファイル準備（存在したら消す）
    child_bins: dict[int, Path] = {}
    child_counts = np.zeros(8, dtype=np.int64)

    for cid in range(8):
        child_key = NodeKey(
            depth=parent_key.depth + 1, path=(parent_key.path << 3) | cid
        )
        child_bin = nodes_dir / f"tmp_d{child_key.depth}_p{child_key.path}.bin"
        child_bin.unlink(missing_ok=True)
        child_bins[cid] = child_bin

    # チャンクごとに読んで、その場で子へ追記
    for start in range(0, n, read_size):
        part = mm[start : start + read_size]
        xs, ys, zs = part["x"], part["y"], part["z"]

        cidx = (
            ((xs >= cx).astype(np.uint8) << 2)
            | ((ys >= cy).astype(np.uint8) << 1)
            | ((zs >= cz).astype(np.uint8) << 0)
        )

        # 子ごとに選んで追記（この時点で copy して memmap 参照を切る）
        for cid in range(8):
            sel = cidx == cid
            if np.any(sel):
                recs = part[sel].copy()
                append_points(child_bins[cid], recs)
                child_counts[cid] += recs.size

        # 参照を早めに切る（Windows対策で効くことがある）
        del part

    # memmap を閉じる→親ファイル削除
    mm._mmap.close()
    del mm
    node_bin.unlink(missing_ok=True)

    # 戻り値を作る（点数0の子は省く）
    out: dict[int, tuple[Path, np.ndarray, np.ndarray, int]] = {}
    for cid in range(8):
        cnt = int(child_counts[cid])
        if cnt <= 0:
            # 何も書いてないならファイルも消しておく
            child_bins[cid].unlink(missing_ok=True)
            continue

        mn, mx = child_aabb(vmin, vmax, cid)
        out[cid] = (child_bins[cid], mn, mx, cnt)

    return out


def child_aabb(vmin, vmax, cidx: int):
    # 最小最大をnumpy配列に変換 [x,y,z]
    vmin = np.asarray(vmin, np.float32)
    vmax = np.asarray(vmax, np.float32)

    # 親のAABBの中心位置
    center = (vmin + vmax) * 0.5

    # 3bitでどこの位置かを
    x = (cidx >> 2) & 1
    y = (cidx >> 1) & 1
    z = (cidx >> 0) & 1
    is_upper = np.array([x, y, z], dtype=np.uint8)

    # 子のAABBの最小最大を取得
    child_min = np.where(is_upper == 0, vmin, center)
    child_max = np.where(is_upper == 0, center, vmax)

    return child_min, child_max


@dataclass(frozen=True)
class NodeKey:
    depth: int
    path: int


def leaf_filename(key: NodeKey) -> str:
    return f"leaf_d{key.depth}_p{key.path}.bin"


def key_to_dict(k: NodeKey) -> dict:
    return {"depth": int(k.depth), "path": int(k.path)}


def dict_to_key(d: dict) -> NodeKey:
    return NodeKey(depth=int(d["depth"]), path=int(d["path"]))


# ------------------------
# 4) LOD構築
# ------------------------
def build_leaf_lods(config: Config) -> None:

    # 参照を取得
    point_dtype = config.options.point_dtype
    lod_steps = config.options.lod_steps
    nodes_dir = config.paths.nodes_dir
    leaf_lods_path = config.paths.leaf_lods_path
    oct_manifest_path = config.paths.oct_manifest_path

    with oct_manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    leaves = manifest["leaves"]
    # 安定順（depth→path）
    leaves_sorted = sorted(
        leaves, key=lambda x: (int(x["key"]["depth"]), int(x["key"]["path"]))
    )

    offset = 0
    point_struct = struct.Struct("<HHHBBBB")

    leaf_records = []

    for leaf in tqdm(leaves_sorted, desc="LOD構築", unit="leaf"):
        key = dict_to_key(leaf["key"])
        leaf_bin = nodes_dir / leaf["file"]

        mm = np.memmap(leaf_bin, mode="r", dtype=point_dtype)
        n = int(mm.shape[0])

        used = np.zeros(n, bool)
        cum = {}

        for s in sorted(lod_steps, reverse=True):
            idx = utils.downsample_indices(n, step=int(s))
            new = idx[~used[idx]]
            used[new] = True
            cum[int(s)] = int(used.sum())

        total_points = cum[int(lod_steps[0])]
        base = offset
        lods = [(base, cum[int(s)]) for s in lod_steps]
        offset += total_points * point_struct.size

        leaf_records.append(
            {
                "key": {"depth": key.depth, "path": key.path},
                "min": leaf["min"],
                "max": leaf["max"],
                "count": int(leaf["count"]),
                "file": leaf["file"],
                "lods": [[int(a), int(b)] for (a, b) in lods],
            }
        )

        mm._mmap.close()
        del mm

    out = {
        "lod_steps": [int(s) for s in lod_steps],
        "point_struct": "<HHHBBBB",
        "leaf_count": len(leaf_records),
        "total_body_bytes": int(offset),
        "leaf_records": leaf_records,
    }

    with leaf_lods_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ------------------------
# 5) OPC出力
# ------------------------
def write_opc(config: Config) -> None:

    # 参照を取得
    magic = config.options.magic
    version_major = config.options.version_major
    version_minor = config.options.version_minor
    version_patch = config.options.version_patch
    lod_steps = config.options.lod_steps
    point_dtype = config.options.point_dtype
    oct_manifest_path = config.paths.oct_manifest_path
    leaf_lods_path = config.paths.leaf_lods_path
    nodes_dir = config.paths.nodes_dir
    opc_path = config.paths.opc_path

    with oct_manifest_path.open("r", encoding="utf-8") as f:
        oct_manifest = json.load(f)
    with leaf_lods_path.open("r", encoding="utf-8") as f:
        lod_data = json.load(f)

    root_min = np.asarray(oct_manifest["root_min"], np.float32)
    root_max = np.asarray(oct_manifest["root_max"], np.float32)

    leaf_records = lod_data["leaf_records"]
    point_struct = struct.Struct("<HHHBBBB")

    # --- ヘッダー ---
    with opc_path.open("wb") as fp:
        fp.write(magic)
        fp.write(struct.pack("<B", version_major))
        fp.write(struct.pack("<B", version_minor))
        fp.write(struct.pack("<B", version_patch))

        # LOD段階数
        fp.write(struct.pack("<B", len(lod_steps)))

        # root AABB
        fp.write(struct.pack("<3f", *root_min))
        fp.write(struct.pack("<3f", *root_max))

        # 葉数
        fp.write(struct.pack("<I", len(leaf_records)))

        # leaf meta: depth(u8), path(u64), aabb, (offset(u64),count(u32))*N
        # ※ path は可変長だが、ここでは u64 に入れて固定長にする（簡単で速い）
        for r in leaf_records:
            depth = int(r["key"]["depth"])
            path = int(r["key"]["path"])
            fp.write(struct.pack("<B", depth))
            fp.write(struct.pack("<Q", path))
            fp.write(struct.pack("<3f", *np.asarray(r["min"], np.float32)))
            fp.write(struct.pack("<3f", *np.asarray(r["max"], np.float32)))
            for off, cnt in r["lods"]:
                fp.write(struct.pack("<Q", int(off)))
                fp.write(struct.pack("<I", int(cnt)))

    # --- ボディ ---
    with opc_path.open("ab") as fp:
        for r in tqdm(leaf_records, desc="OPCボディ出力", unit="leaf"):
            leaf_bin = nodes_dir / r["file"]
            mm = np.memmap(leaf_bin, mode="r", dtype=point_dtype)
            n = int(mm.shape[0])

            used = np.zeros(n, bool)
            write_parts = []
            for s in sorted(lod_steps, reverse=True):
                idx = utils.downsample_indices(n, step=int(s))
                new = idx[~used[idx]]
                used[new] = True
                write_parts.append(new)
            order = (
                np.concatenate(write_parts) if write_parts else np.array([], np.int64)
            )

            vmin = np.asarray(r["min"], np.float32)
            vmax = np.asarray(r["max"], np.float32)

            qx = utils.quantize_u16(mm["x"][order], vmin[0], vmax[0])
            qy = utils.quantize_u16(mm["y"][order], vmin[1], vmax[1])
            qz = utils.quantize_u16(mm["z"][order], vmin[2], vmax[2])

            rr = utils.convert_color(mm["r"][order])
            gg = utils.convert_color(mm["g"][order])
            bb = utils.convert_color(mm["b"][order])
            aa = np.full_like(rr, 255, dtype=np.uint8)

            for x16, y16, z16, r8, g8, b8, a8 in zip(qx, qy, qz, rr, gg, bb, aa):
                fp.write(
                    point_struct.pack(
                        int(x16), int(y16), int(z16), int(r8), int(g8), int(b8), int(a8)
                    )
                )

            mm._mmap.close()
            del mm


# ------------------------
# 4) 中間ファイル削除
# ------------------------
def cleanup_intermediate_files(config: Config) -> None:
    # 参照を取得
    points_cache_path = config.paths.points_path
    oct_manifest_path = config.paths.oct_manifest_path
    leaf_lods_path = config.paths.leaf_lods_path
    nodes_dir = config.paths.nodes_dir

    # points.bin
    if points_cache_path.exists():
        points_cache_path.unlink()

    # octree_manifest.json
    if oct_manifest_path.exists():
        oct_manifest_path.unlink()

    # leaf_lods.json
    if leaf_lods_path.exists():
        leaf_lods_path.unlink()

    # nodes フォルダ（中身ごと）
    if nodes_dir.exists() and nodes_dir.is_dir():
        shutil.rmtree(nodes_dir)


# ------------------------
# メイン処理
# ------------------------
def main() -> None:
    LAS_DIR = Path(
        r"C:\Users\user\Github\PointCloudViewer\Assets\PointCloudViewer\Las"
    )
    OUT_DIR = Path(r"C:\Users\user\Desktop")

    paths = Paths(las_dir=LAS_DIR, out_dir=OUT_DIR)
    options = Options()
    config = Config(paths=paths, options=options)

    global_center = compute_global_center(config)
    export_las_to_points_bin(config, global_center)
    build_octree(config)
    build_leaf_lods(config)
    write_opc(config)
    cleanup_intermediate_files(config)


if __name__ == "__main__":
    main()
