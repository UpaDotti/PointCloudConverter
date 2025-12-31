# 標準ライブラリ
import json
import struct
import sys
from io import BytesIO
from pathlib import Path
import shutil

# サードパーティー
import laspy
import numpy as np
from tqdm import tqdm

# 自作モジュール
from PointCloudConverter import utils

# ------------------------
# 設定
# ------------------------
LAS_DIR = Path(r"C:\Users\user\Github\PointCloudViewer\Assets\PointCloudViewer\Las")
OUT_DIR = Path(r"C:\Users\user\Desktop")

TPC_FILE_NAME = "sample.tpc"
TPC_PATH = OUT_DIR / TPC_FILE_NAME

POINTS_CACHE_FILENAME = "points.bin"
POINTS_CACHE_PATH = OUT_DIR / POINTS_CACHE_FILENAME

CHUNK_DIR_NAME = "chunks"
CHUNK_DIR = OUT_DIR / CHUNK_DIR_NAME

MANIFEST_FILENAME = "chunks_manifest.json"
MANIFEST_PATH = OUT_DIR / MANIFEST_FILENAME

MAGIC = b"TPC\x00"
VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_PATCH = 0
GRID_SIZE = 100.0

LOD_LEVELS = 7
LOD_FIRST_STEP = 1
LOD_STEP_FACTOR = 4
LOD_STEPS = ()

READ_SIZE = 50_000
PATTERN = "*.las"
POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("r", "<u2"),
        ("g", "<u2"),
        ("b", "<u2"),
    ]
)


# ------------------------
# LODステップを生成
# ------------------------
LOD_STEPS = utils.make_lod_steps(LOD_LEVELS, LOD_FIRST_STEP, LOD_STEP_FACTOR)


# ------------------------
# グローバル中心計算
# ------------------------
def compute_global_center():

    las_paths = list(LAS_DIR.glob(PATTERN))

    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0

    for las_path in tqdm(las_paths, desc="グローバル中心", unit="file"):
        with laspy.open(las_path) as reader:
            scale = np.array(reader.header.scales, dtype=np.float64)
            offset = np.array(reader.header.offsets, dtype=np.float64)

            # チャンク単位でも細かい進捗を出したければ、内側にも tqdm を巻ける
            for points in reader.chunk_iterator(READ_SIZE):
                n = points["X"].size
                sum_x += scale[0] * points["X"].sum(dtype=np.int64) + n * offset[0]
                sum_y += (
                    scale[2] * points["Z"].sum(dtype=np.int64) + n * offset[2]
                )  # Z→Y
                sum_z += (
                    scale[1] * points["Y"].sum(dtype=np.int64) + n * offset[1]
                )  # Y→Z
                count += n

    center_f64 = np.array([sum_x, sum_y, sum_z], dtype=np.float64) / count
    center_f32 = center_f64.astype(np.float32)
    return center_f32


global_center = compute_global_center()


# ------------------------
# 点データを一時キャッシュとして書き出し
# ------------------------
def export_las_points(global_center):

    las_paths = list(LAS_DIR.glob(PATTERN))

    total = 0
    with POINTS_CACHE_PATH.open("wb") as fp:
        for las_path in tqdm(las_paths, desc="点群キャッシュ", unit="file"):
            with laspy.open(las_path) as reader:
                scale = np.array(reader.header.scales, dtype=np.float32)
                offset = np.array(reader.header.offsets, dtype=np.float32)

                for pts in reader.chunk_iterator(READ_SIZE):
                    # 実座標化
                    x = pts["X"].astype(np.float32, copy=False) * scale[0] + offset[0]
                    y_real = (
                        pts["Y"].astype(np.float32, copy=False) * scale[1] + offset[1]
                    )
                    z_real = (
                        pts["Z"].astype(np.float32, copy=False) * scale[2] + offset[2]
                    )

                    # Y/Zを入替
                    y, z = z_real, y_real

                    # センタリング
                    x -= global_center[0]
                    y -= global_center[1]
                    z -= global_center[2]

                    # RGB16
                    r = pts["red"].astype(np.uint16, copy=False)
                    g = pts["green"].astype(np.uint16, copy=False)
                    b = pts["blue"].astype(np.uint16, copy=False)

                    # まとめて書き出し
                    recs = np.empty(x.size, dtype=POINT_DTYPE)
                    recs["x"] = x
                    recs["y"] = y
                    recs["z"] = z
                    recs["r"] = r
                    recs["g"] = g
                    recs["b"] = b

                    recs.tofile(fp)
                    total += x.size


export_las_points(global_center)


# ------------------------
# チャンク分割
# ------------------------
def chunk_point_cloud():

    # 点の数を計算
    size = POINTS_CACHE_PATH.stat().st_size
    itemsize = POINT_DTYPE.itemsize
    total = size // itemsize

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    meta = {}
    processed = 0

    with POINTS_CACHE_PATH.open("rb") as fp, tqdm(
        total=total, desc="チャンク分割", unit="pts"
    ) as pbar:
        while processed < total:
            # 一気に読み込むサイズを制限
            n = min(READ_SIZE, total - processed)

            # 点データを取得
            recs = np.fromfile(fp, dtype=POINT_DTYPE, count=n)

            if recs.size == 0:
                break

            # 処理済みの点をカウント
            processed += recs.size
            pbar.update(recs.size)

            # XYZそれぞれで取得
            x = recs["x"]
            y = recs["y"]
            z = recs["z"]

            # 各点の座標をグリッド座標に変換
            ix = np.floor_divide(x, GRID_SIZE).astype(np.int32)
            iy = np.floor_divide(y, GRID_SIZE).astype(np.int32)
            iz = np.floor_divide(z, GRID_SIZE).astype(np.int32)

            # グリッド座標をキーにする
            keys = np.stack([ix, iy, iz], axis=1)

            # 重複なしキー 各点がどのuniqを使用するかのインデックス
            uniq_keys, point_ids = np.unique(keys, axis=0, return_inverse=True)

            for index, key in enumerate(uniq_keys):
                # グリッドごとの点を取得
                select = point_ids == index
                part = recs[select]

                ixk, iyk, izk = map(int, key)
                chunk_path = CHUNK_DIR / f"chunk_{ixk}_{iyk}_{izk}.bin"
                with chunk_path.open("ab") as cf:
                    part.tofile(cf)

                xyz = np.stack([part["x"], part["y"], part["z"]], axis=1)
                xyz_min = xyz.min(axis=0)
                xyz_max = xyz.max(axis=0)
                count = int(part.size)

                if (ixk, iyk, izk) not in meta:
                    meta[(ixk, iyk, izk)] = {
                        "min": xyz_min.tolist(),
                        "max": xyz_max.tolist(),
                        "count": count,
                    }
                else:
                    m = meta[(ixk, iyk, izk)]
                    m["min"] = np.minimum(
                        np.array(m["min"], np.float32), xyz_min
                    ).tolist()
                    m["max"] = np.maximum(
                        np.array(m["max"], np.float32), xyz_max
                    ).tolist()
                    m["count"] += count

    chunks = []
    for k, v in sorted(meta.items()):
        ix, iy, iz = k
        chunks.append(
            {
                "grid_size": float(GRID_SIZE),
                "ix": ix,
                "iy": iy,
                "iz": iz,
                "min": v["min"],
                "max": v["max"],
                "count": v["count"],
                "file": f"chunk_{ix}_{iy}_{iz}.bin",
            }
        )

    manifest = {
        "grid_size": float(GRID_SIZE),
        "chunks": chunks,
    }

    with (MANIFEST_PATH).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


chunk_point_cloud()


# ------------------------
# LODごとのオフセットを計算
# ------------------------


def build_lods_meta_only() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    sorted_chunks = sorted(
        manifest["chunks"],
        key=lambda c: (int(c["ix"]), int(c["iy"]), int(c["iz"])),
    )

    chunk_records: list[dict] = []

    offset = 0
    point_struct = struct.Struct("<HHHBBBB")  # x,y,z(ushort), r,g,b,a(uint8)

    for meta in tqdm(sorted_chunks, desc="LODメタ計算", unit="chunk"):
        chunk_path = CHUNK_DIR / meta["file"]

        mm = np.memmap(chunk_path, mode="r", dtype=POINT_DTYPE)
        n = mm.shape[0]

        used = np.zeros(n, bool)
        cum: dict[int, int] = {}

        for s in sorted(LOD_STEPS, reverse=True):
            idx = utils.downsample_indices(n, step=int(s))
            new = idx[~used[idx]]
            used[new] = True
            cum[int(s)] = int(used.sum())

        # このチャンクで実際に書き出される総ポイント数（=最も細かいLODの点数）
        total_points = cum[int(LOD_STEPS[0])]

        # このチャンク内でのボディ先頭からのオフセット
        chunk_base_offset = offset

        # 各LODの (offset, count) を記録
        lods = [(chunk_base_offset, cum[int(s)]) for s in LOD_STEPS]

        # 次のチャンク用にオフセットを進める
        offset += total_points * point_struct.size

        chunk_records.append(
            {
                "ix": int(meta["ix"]),
                "iy": int(meta["iy"]),
                "iz": int(meta["iz"]),
                "min": meta["min"],
                "max": meta["max"],
                "lods": lods,
                "file": meta["file"],
            }
        )

        del mm

    return chunk_records


chunk_records = build_lods_meta_only()


# ------------------------
# TCPのヘッダーを書き込み
# ------------------------


def write_tpc_header(chunk_records: list[dict]) -> None:
    with TPC_PATH.open("wb") as fp:
        # マジック＋バージョン
        fp.write(MAGIC)  # b"TPC\x00"
        fp.write(struct.pack("<B", VERSION_MAJOR))
        fp.write(struct.pack("<B", VERSION_MINOR))
        fp.write(struct.pack("<B", VERSION_PATCH))

        # LOD段階数
        fp.write(struct.pack("<B", len(LOD_STEPS)))
        # グリッドサイズ
        fp.write(struct.pack("<f", GRID_SIZE))
        # チャンク数
        fp.write(struct.pack("<I", len(chunk_records)))
        # 点群のグローバル中心
        fp.write(struct.pack("<3f", *global_center))

        # チャンクごとのメタ情報＋LOD情報
        for record in chunk_records:
            ix = record["ix"]
            iy = record["iy"]
            iz = record["iz"]
            vmin = record["min"]
            vmax = record["max"]
            lods = record["lods"]

            # チャンク座標
            fp.write(struct.pack("<3i", ix, iy, iz))

            # チャンク内のAABB
            fp.write(struct.pack("<3f", *vmin))
            fp.write(struct.pack("<3f", *vmax))

            # 各LODの (offset, count)
            for offset, count in lods:
                fp.write(struct.pack("<Q", int(offset)))  # オフセット（ボディ先頭から）
                fp.write(struct.pack("<I", int(count)))  # 点数


write_tpc_header(chunk_records)
print("ヘッダー書き込み")


# ------------------------
# 点群の本体データを書き込み
# ------------------------


def write_lods_binary(chunk_records: list[dict]) -> None:
    point_struct = struct.Struct("<HHHBBBB")

    with TPC_PATH.open("ab") as fp:
        for record in tqdm(chunk_records, desc="点群バイナリ書き込み", unit="chunk"):
            chunk_path = CHUNK_DIR / record["file"]

            mm = np.memmap(chunk_path, mode="r", dtype=POINT_DTYPE)
            n = mm.shape[0]

            used = np.zeros(n, bool)
            write_parts = []

            for s in sorted(LOD_STEPS, reverse=True):
                idx = utils.downsample_indices(n, step=int(s))
                new = idx[~used[idx]]
                used[new] = True
                write_parts.append(new)

            order = np.concatenate(write_parts)

            vmin = np.array(record["min"], dtype=np.float32)
            vmax = np.array(record["max"], dtype=np.float32)

            qx = utils.quantize_u16(mm["x"][order], vmin[0], vmax[0])
            qy = utils.quantize_u16(mm["y"][order], vmin[1], vmax[1])
            qz = utils.quantize_u16(mm["z"][order], vmin[2], vmax[2])

            r = utils.convert_color(mm["r"][order])
            g = utils.convert_color(mm["g"][order])
            b = utils.convert_color(mm["b"][order])
            a = np.full_like(r, 255, dtype=np.uint8)

            for x16, y16, z16, r8, g8, b8, a8 in zip(qx, qy, qz, r, g, b, a):
                fp.write(
                    point_struct.pack(
                        int(x16),
                        int(y16),
                        int(z16),
                        int(r8),
                        int(g8),
                        int(b8),
                        int(a8),
                    )
                )

            del mm


write_lods_binary(chunk_records)
print("点群バイナリ出力")

# ------------------------
# TPCファイル書き出し
# ------------------------


def cleanup_intermediate_files() -> None:

    # points.bin
    if POINTS_CACHE_PATH.exists():
        POINTS_CACHE_PATH.unlink()

    # manifest
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    # chunks フォルダ（中身ごと）
    if CHUNK_DIR.exists() and CHUNK_DIR.is_dir():
        shutil.rmtree(CHUNK_DIR)


cleanup_intermediate_files()
print("中間ファイル削除")
