import json
import struct
import sys
from io import BytesIO
from pathlib import Path

import laspy
import numpy as np

import utils

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
LOD_STEPS = (1, 4, 16, 64, 256)

READ_SIZE = 10000
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
# グローバル中心計算（1パス目）
# ------------------------
def compute_global_center():

    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    for las_path in LAS_DIR.glob(PATTERN):
        with laspy.open(las_path) as reader:
            scale = np.array(reader.header.scales, dtype=np.float64)
            offset = np.array(reader.header.offsets, dtype=np.float64)

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
print("全体中心の計算 終了")


# ------------------------
# 点データを一時キャッシュとして書き出し（2パス目）
# ------------------------
def export_las_points(global_center):

    total = 0
    with POINTS_CACHE_PATH.open("wb") as fp:
        for las_path in LAS_DIR.glob(PATTERN):
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
print("点群データの書き出し 終了")


# ------------------------
# チャンク分割（3パス目）
# ------------------------
def chunk_point_cloud():

    # 点の数を計算
    size = POINTS_CACHE_PATH.stat().st_size
    itemsize = POINT_DTYPE.itemsize
    total = size // itemsize

    # チャックデータを保存するフォルダを作成
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    meta = {}
    processed = 0
    with POINTS_CACHE_PATH.open("rb") as fp:
        while processed < total:
            # 一気に読み込むサイズを制限
            n = min(10000, total - processed)

            # 点データを取得
            recs = np.fromfile(fp, dtype=POINT_DTYPE, count=n)

            print(processed / total)

            # 点が空の場合はループを終了
            if recs.size == 0:
                break

            # 処理済みの点をカウント
            processed += recs.size

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

                # 点データを書き出し
                ixk, iyk, izk = map(int, key)
                chunk_path = CHUNK_DIR / f"chunk_{ixk}_{iyk}_{izk}.bin"
                with chunk_path.open("ab") as cf:
                    part.tofile(cf)

                # グリッド点群の最小/最大/数を取得
                xyz = np.stack([part["x"], part["y"], part["z"]], axis=1)
                xyz_min = xyz.min(axis=0)
                xyz_max = xyz.max(axis=0)
                count = int(part.size)

                # メタデータを更新
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

            # チャンクデータの配列を作成
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

            # マニュフェストを作成
            manifest = {
                "grid_size": float(GRID_SIZE),
                "chunks": chunks,
            }

    with (MANIFEST_PATH).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


chunk_point_cloud()
print("チャック分割 終了")

# ------------------------
# LOD構築（4パス目）
# ------------------------


def build_lods() -> tuple[list[dict], bytes]:
    # MANIFEST読み込み
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # ix, iy, iz でソート
    sorted_chunks = sorted(
        manifest["chunks"],
        key=lambda c: (int(c["ix"]), int(c["iy"]), int(c["iz"])),
    )

    # 出力用バッファとメタ配列
    chunk_records: list[dict] = []
    body_buffer = BytesIO()
    offset = 0

    point_struct = struct.Struct("<HHHBBBB")  # x,y,z(ushort), r,g,b,a(uint8)

    for meta in sorted_chunks:
        ix = int(meta["ix"])
        iy = int(meta["iy"])
        iz = int(meta["iz"])
        vmin = np.array(meta["min"], dtype=np.float32)
        vmax = np.array(meta["max"], dtype=np.float32)
        chunk_path = CHUNK_DIR / meta["file"]

        # チャンクをメモリマップで読む
        mm = np.memmap(chunk_path, mode="r", dtype=POINT_DTYPE)
        n = mm.shape[0]

        # LOD降順で並び順を作る
        used = np.zeros(n, bool)  # 点を採用したかを管理するフラグ
        write_parts = []  # 書き出し最終順序
        cum: dict[int, int] = {}  # LODごとの累積点数

        for s in sorted(LOD_STEPS, reverse=True):
            idx = utils.downsample_indices(n, step=int(s))  # インデックス候補
            new = idx[~used[idx]]  # 採用していないインデックスを抽出
            write_parts.append(new)  # 書き出しに追加
            used[new] = True  # 採用を記録
            cum[int(s)] = int(used.sum())  # 累積点数を記録

        # LODごとになっているインデックスをまとめる
        order = np.concatenate(write_parts)

        # 位置を量子化
        qx = utils.quantize_u16(mm["x"][order], vmin[0], vmax[0])
        qy = utils.quantize_u16(mm["y"][order], vmin[1], vmax[1])
        qz = utils.quantize_u16(mm["z"][order], vmin[2], vmax[2])

        # 色を変換
        r = utils.convert_color(mm["r"][order])
        g = utils.convert_color(mm["g"][order])
        b = utils.convert_color(mm["b"][order])
        a = np.full_like(r, 255, dtype=np.uint8)

        # このチャンクの点データが始まるバイト位置
        chunk_base_offset = offset

        # 点群バッファに記録
        for x16, y16, z16, r8, g8, b8, a8 in zip(qx, qy, qz, r, g, b, a):
            point_pack = point_struct.pack(
                int(x16),
                int(y16),
                int(z16),
                int(r8),
                int(g8),
                int(b8),
                int(a8),
            )
            body_buffer.write(point_pack)

        # チャンクごとのオフセットを進める
        offset += len(order) * point_struct.size

        # chunk_base_offset：チャンク先頭 cumulative_counts：先頭から何点読むか
        lods = [(chunk_base_offset, cum[s]) for s in LOD_STEPS]

        # チャンクデータを追加
        chunk_records.append(
            {
                "ix": ix,
                "iy": iy,
                "iz": iz,
                "min": vmin.tolist(),
                "max": vmax.tolist(),
                "lods": lods,
            }
        )

        # 早めに参照を切っておきたい場合は明示的に消す
        del mm

    # body_bufferの中身（bytes）を返す
    return chunk_records, body_buffer.getvalue()


chunk_records, body_bytes = build_lods()
print("LOD構築 終了")

# ------------------------
# TPCファイル書き出し
# ------------------------

# ファイル全体
with TPC_PATH.open("wb") as fp:
    fp.write(MAGIC)  # マジックナンバー
    fp.write(struct.pack("<B", VERSION_MAJOR))  # バージョン（メジャー）
    fp.write(struct.pack("<B", VERSION_MINOR))  # バージョン（マイナー）
    fp.write(struct.pack("<B", VERSION_PATCH))  # バージョン（パッチ）
    fp.write(struct.pack("<B", len(LOD_STEPS)))  # LOD
    fp.write(struct.pack("<f", GRID_SIZE))  # グリッドサイズ
    fp.write(struct.pack("<I", len(chunk_records)))  # チャンクデータ
    fp.write(struct.pack("<3f", *global_center))  # 点群の中心

    # チャンクデータ
    for record in chunk_records:
        fp.write(
            # チャンク座標 ex :(1,1,1)
            struct.pack("<3i", record["ix"], record["iy"], record["iz"])
        )
        fp.write(struct.pack("<3f", *record["min"]))  # チャンク内での最小
        fp.write(struct.pack("<3f", *record["max"]))  # チャンク内での最大

        # LOD別データ
        for offset, count in record["lods"]:
            fp.write(struct.pack("<Q", offset))  # オフセット…開始位置
            fp.write(struct.pack("<I", count))  # 点の数

    fp.write(body_bytes)  # 点データ

print("✅ 書き出し完了: TPCファイル出力成功")
