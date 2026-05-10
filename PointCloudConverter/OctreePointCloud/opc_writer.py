import json
import struct
import numpy as np
from tqdm import tqdm

from PointCloudConverter import utils
from .config import Config


def write_opc(config: Config) -> None:
    """OPCファイルを出力"""

    magic = config.options.magic
    version_major = config.options.version_major
    version_minor = config.options.version_minor
    version_patch = config.options.version_patch
    lod_steps = config.options.lod_steps
    in_dtype = config.options.in_dtype
    out_dtype = config.options.out_dtype
    oct_manifest_path = config.paths.oct_manifest_path
    leaf_lods_path = config.paths.leaf_lods_path
    nodes_dir = config.paths.nodes_dir
    opc_path = config.paths.opc_path

    # オクトツリーマニュフェストを読込
    with oct_manifest_path.open("r", encoding="utf-8") as f:
        oct_manifest = json.load(f)

    # branchデータを取得
    branches = oct_manifest["branches"]

    # LODマニュフェストを読込
    with leaf_lods_path.open("r", encoding="utf-8") as f:
        lod_data = json.load(f)

    # LODのleafデータを取得
    leaf_records = lod_data["leaf_records"]

    #  サイズを計算
    root_min = np.array(oct_manifest["root_min"], dtype=np.float32)
    root_max = np.array(oct_manifest["root_max"], dtype=np.float32)
    size = root_max - root_min

    # --- ヘッダー ---
    with opc_path.open("wb") as fp:
        # Version
        fp.write(magic)
        fp.write(struct.pack("<B", version_major))
        fp.write(struct.pack("<B", version_minor))
        fp.write(struct.pack("<B", version_patch))

        # Bound
        fp.write(struct.pack("<3f", *size))
        fp.write(struct.pack("<3f", *root_min))
        fp.write(struct.pack("<3f", *root_max))

        # LOD
        fp.write(struct.pack("<B", len(lod_steps)))
        for s in lod_steps:
            fp.write(struct.pack("<I", int(s)))

        # Status
        fp.write(struct.pack("<I", len(branches)))
        fp.write(struct.pack("<I", len(leaf_records)))
        fp.write(struct.pack("<Q", oct_manifest["total_points"]))
        fp.write(struct.pack("<H", int(out_dtype.itemsize)))

        # branchのメタデータ
        for branch in branches:
            fp.write(struct.pack("<B", branch["key"]["depth"]))
            fp.write(struct.pack("<Q", branch["key"]["path"]))
            fp.write(struct.pack("<3f", *branch["min"]))
            fp.write(struct.pack("<3f", *branch["max"]))

        # leafのメタデータ
        for leaf_idx, record in enumerate(leaf_records):
            fp.write(struct.pack("<I", leaf_idx))
            fp.write(struct.pack("<B", record["key"]["depth"]))
            fp.write(struct.pack("<Q", record["key"]["path"]))
            fp.write(struct.pack("<3f", *record["min"]))
            fp.write(struct.pack("<3f", *record["max"]))
            fp.write(struct.pack("<Q", record["base_offset"]))

            # LODごとの点数
            for c in record["lod_counts"]:
                fp.write(struct.pack("<I", c))

    # --- ボディ ---
    lod_steps_desc = sorted(lod_steps, reverse=True)
    with opc_path.open("ab") as fp:

        # leafループ
        for record in tqdm(leaf_records, desc="OPCボディ出力", unit="leaf"):

            leaf_bin = nodes_dir / record["file"]
            mm = np.memmap(leaf_bin, mode="r", dtype=in_dtype)
            n = int(mm.shape[0])
            used = np.zeros(n, bool)

            # 書き出し順
            write_parts = []

            # LODループ
            for step in lod_steps_desc:
                idx = utils.downsample_indices(n, step=int(step))
                new = idx[~used[idx]]
                used[new] = True
                write_parts.append(new)

            # LODごとに分かれている配列を連結
            order = np.concatenate(write_parts)

            # leafの最小 / 最大
            vmin = record["min"]
            vmax = record["max"]

            # 座標を量子化
            qx = utils.quantize_u16(mm["x"][order], vmin[0], vmax[0])
            qy = utils.quantize_u16(mm["y"][order], vmin[1], vmax[1])
            qz = utils.quantize_u16(mm["z"][order], vmin[2], vmax[2])

            # 色を量子化
            rr = utils.convert_color(mm["r"][order])
            gg = utils.convert_color(mm["g"][order])
            bb = utils.convert_color(mm["b"][order])

            # leafの点データを構成
            out = np.empty(order.size, dtype=out_dtype)
            out["x"] = qx.astype(np.uint16, copy=False)
            out["y"] = qy.astype(np.uint16, copy=False)
            out["z"] = qz.astype(np.uint16, copy=False)
            out["r"] = rr.astype(np.uint8, copy=False)
            out["g"] = gg.astype(np.uint8, copy=False)
            out["b"] = bb.astype(np.uint8, copy=False)
            out["a"] = 255

            out.tofile(fp)

            mm._mmap.close()
            del mm
