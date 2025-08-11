import laspy
import numpy as np
from pathlib import Path
import struct
from io import BytesIO

# ------------------------
# 設定
# ------------------------
las_folder = Path(r"C:\Users\user\Github\PointCloudViewer\Assets\PointCloudViewer\Las")
out_path = Path(r"C:\Users\user\Desktop\sample.tpc")
magic = b"TPC\x00"
version_major = 1
version_minor = 0
version_patch = 0
grid_size = 100.0
lod_steps = (1, 4, 16)


# ------------------------
# 関数
# ------------------------
def gamma_to_linear(c):
    # sRGB(ガンマ)値をリニアRGBに変換
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def downsample_indices(n, step):
    # 0〜n-1 のインデックスを step 間隔で取得
    return np.arange(0, n, step, dtype=np.int32)


# ------------------------
# グローバル中心計算（1パス目）
# ------------------------
all_xyz = []

for las_path in las_folder.glob("*.las"):
    las = laspy.read(las_path)
    scale = np.array(las.header.scales, dtype=np.float32)
    offset = np.array(las.header.offsets, dtype=np.float32)

    x = las.points["X"] * scale[0] + offset[0]
    y = las.points["Y"] * scale[1] + offset[1]
    z = las.points["Z"] * scale[2] + offset[2]
    y, z = z, y  # yとzを入替

    all_xyz.append(np.stack([x, y, z], axis=1))

global_xyz = np.concatenate(all_xyz, axis=0)
global_center = global_xyz.mean(axis=0).astype(np.float32)
print(f"全体中心: {global_center}")

# ------------------------
# チャンク分割（2パス目）
# ------------------------
chunks = {}
total_point_count = 0

for las_path in las_folder.glob("*.las"):
    las = laspy.read(las_path)
    scale = np.array(las.header.scales, dtype=np.float32)
    offset = np.array(las.header.offsets, dtype=np.float32)

    x = las.points["X"] * scale[0] + offset[0]
    y = las.points["Y"] * scale[1] + offset[1]
    z = las.points["Z"] * scale[2] + offset[2]
    y, z = z, y  # yとzを入替

    # 中心からの相対座標に変換
    x -= global_center[0]
    y -= global_center[1]
    z -= global_center[2]

    # 色処理
    r_linear = gamma_to_linear((las.red / 65535.0) * 255.0)
    g_linear = gamma_to_linear((las.green / 65535.0) * 255.0)
    b_linear = gamma_to_linear((las.blue / 65535.0) * 255.0)

    r = (r_linear * 255.0).astype(np.uint8)
    g = (g_linear * 255.0).astype(np.uint8)
    b = (b_linear * 255.0).astype(np.uint8)

    # チャンクごとにグループ分け
    for i in range(len(x)):
        ix = int(x[i] // grid_size)
        iy = int(y[i] // grid_size)
        iz = int(z[i] // grid_size)
        key = (ix, iy, iz)

        if key not in chunks:
            chunks[key] = {
                "points": [],
                "min": np.array([x[i], y[i], z[i]], dtype=np.float32),
                "max": np.array([x[i], y[i], z[i]], dtype=np.float32),
            }

        pt = (x[i], y[i], z[i], r[i], g[i], b[i])
        chunks[key]["points"].append(pt)

        pos = np.array([x[i], y[i], z[i]], dtype=np.float32)
        chunks[key]["min"] = np.minimum(chunks[key]["min"], pos)
        chunks[key]["max"] = np.maximum(chunks[key]["max"], pos)

        total_point_count += 1


print(f"チャンク数: {len(chunks)}")
print(f"全体点数: {total_point_count}")

# ------------------------
# LOD構築（3パス目）
# ------------------------

chunk_records = []
body_buffer = BytesIO()
offset = 0

pos_struct = struct.Struct("<3f")
col_struct = struct.Struct("<3B")
point_bytes = pos_struct.size + col_struct.size

for key, data in chunks.items():
    ix, iy, iz = key
    pts = data["points"]
    n = len(pts)

    lods = []
    for step in lod_steps:
        idx = downsample_indices(n, step=step)
        lod_offset = offset

        for i in idx:
            x, y, z, r, g, b = pts[i]
            body_buffer.write(pos_struct.pack(x, y, z))
            body_buffer.write(col_struct.pack(r, g, b))

        lod_count = len(idx)
        lods.append((lod_offset, lod_count))
        offset += lod_count * point_bytes

    chunk_records.append(
        {
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "min": data["min"],
            "max": data["max"],
            "lods": lods,
        }
    )

# ------------------------
# TPCファイル書き出し
# ------------------------
with out_path.open("wb") as fp:
    fp.write(magic)
    fp.write(struct.pack("<B", version_major))
    fp.write(struct.pack("<B", version_minor))
    fp.write(struct.pack("<B", version_patch))
    fp.write(struct.pack("<B", len(lod_steps)))
    fp.write(struct.pack("<f", grid_size))
    fp.write(struct.pack("<I", len(chunk_records)))

    fp.write(struct.pack("<3f", *global_center))

    for rec in chunk_records:
        fp.write(struct.pack("<3i", rec["ix"], rec["iy"], rec["iz"]))
        fp.write(struct.pack("<3f", *rec["min"]))
        fp.write(struct.pack("<3f", *rec["max"]))
        for off, cnt in rec["lods"]:
            fp.write(struct.pack("<Q", off))
            fp.write(struct.pack("<I", cnt))

    fp.write(body_buffer.getvalue())

print("✅ 書き出し完了: TPCファイル出力成功")
