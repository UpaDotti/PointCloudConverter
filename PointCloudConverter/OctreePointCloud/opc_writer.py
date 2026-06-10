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
    in_dtype = config.options.in_dtype
    out_dtype = config.options.out_dtype
    oct_manifest_path = config.paths.oct_manifest_path
    leaf_lods_path = config.paths.leaf_lods_path
    branch_lods_path = config.paths.branch_lods_path
    nodes_dir = config.paths.nodes_dir
    opc_path = config.paths.opc_path

    # オクトツリーマニフェストを読込
    with oct_manifest_path.open("r", encoding="utf-8") as f:
        oct_manifest = json.load(f)

    # LODマニフェストを読込
    with leaf_lods_path.open("r", encoding="utf-8") as f:
        lod_data = json.load(f)

    # LODのleafデータを取得
    leaf_records = lod_data["leaf_records"]

    # 枝の代表点メタを読込
    with branch_lods_path.open("r", encoding="utf-8") as f:
        branch_lod_data = json.load(f)
    branch_records = branch_lod_data["branch_records"]

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

        # Status
        fp.write(struct.pack("<I", len(branch_records)))
        fp.write(struct.pack("<I", len(leaf_records)))
        fp.write(struct.pack("<Q", oct_manifest["total_points"]))
        fp.write(struct.pack("<H", int(out_dtype.itemsize)))

        # branchのメタデータ
        for record in branch_records:
            fp.write(struct.pack("<B", record["key"]["depth"]))
            fp.write(struct.pack("<Q", record["key"]["path"]))
            fp.write(struct.pack("<3f", *record["min"]))
            fp.write(struct.pack("<3f", *record["max"]))
            fp.write(struct.pack("<Q", int(record["base_offset"])))
            fp.write(struct.pack("<I", int(record["point_count"])))

        # leafのメタデータ
        for leaf_idx, record in enumerate(leaf_records):
            fp.write(struct.pack("<I", leaf_idx))
            fp.write(struct.pack("<B", record["key"]["depth"]))
            fp.write(struct.pack("<Q", record["key"]["path"]))
            fp.write(struct.pack("<3f", *record["min"]))
            fp.write(struct.pack("<3f", *record["max"]))
            fp.write(struct.pack("<Q", record["base_offset"]))
            fp.write(struct.pack("<I", int(record["point_count"])))

    # 描画時に「先頭N点」で均等分布になるよう順序をシャッフルする乱数源
    rng = np.random.default_rng(seed=0)

    # --- ボディ ---
    with opc_path.open("ab") as fp:

        # leafループ
        for record in tqdm(leaf_records, desc="OPCボディ出力(Leaf)", unit="leaf"):

            leaf_bin = nodes_dir / record["file"]
            mm = np.memmap(leaf_bin, mode="r", dtype=in_dtype)
            n = int(mm.shape[0])

            # leafの最小 / 最大
            vmin = record["min"]
            vmax = record["max"]

            # 座標を量子化
            qx = utils.quantize_u16(mm["x"], vmin[0], vmax[0])
            qy = utils.quantize_u16(mm["y"], vmin[1], vmax[1])
            qz = utils.quantize_u16(mm["z"], vmin[2], vmax[2])

            # 色を量子化
            rr = utils.convert_color(mm["r"])
            gg = utils.convert_color(mm["g"])
            bb = utils.convert_color(mm["b"])

            # leafの点データを構成
            out = np.empty(n, dtype=out_dtype)
            out["x"] = qx.astype(np.uint16, copy=False)
            out["y"] = qy.astype(np.uint16, copy=False)
            out["z"] = qz.astype(np.uint16, copy=False)
            out["r"] = rr.astype(np.uint8, copy=False)
            out["g"] = gg.astype(np.uint8, copy=False)
            out["b"] = bb.astype(np.uint8, copy=False)
            out["a"] = 255

            # 描画時の動的間引きで偏りが出ないよう順序をシャッフル
            if n > 1:
                rng.shuffle(out)

            out.tofile(fp)

            mm._mmap.close()
            del mm

            # 書き出し済みの葉binは以降使わないため削除（ピーク使用量を削減）
            leaf_bin.unlink(missing_ok=True)

        # branchループ
        for record in tqdm(branch_records, desc="OPCボディ出力(Branch)", unit="branch"):

            # 代表点が無い枝はスキップ
            if int(record["point_count"]) == 0 or record["file"] is None:
                continue

            # 代表点のbinをmemmapで開く
            branch_bin = nodes_dir / record["file"]
            mm = np.memmap(branch_bin, mode="r", dtype=in_dtype)
            n = int(mm.shape[0])

            # 量子化用のbounds
            vmin = record["min"]
            vmax = record["max"]

            # 座標を量子化
            qx = utils.quantize_u16(mm["x"], vmin[0], vmax[0])
            qy = utils.quantize_u16(mm["y"], vmin[1], vmax[1])
            qz = utils.quantize_u16(mm["z"], vmin[2], vmax[2])

            # 色を量子化
            rr = utils.convert_color(mm["r"])
            gg = utils.convert_color(mm["g"])
            bb = utils.convert_color(mm["b"])

            # 出力配列を構成
            out = np.empty(n, dtype=out_dtype)
            out["x"] = qx.astype(np.uint16, copy=False)
            out["y"] = qy.astype(np.uint16, copy=False)
            out["z"] = qz.astype(np.uint16, copy=False)
            out["r"] = rr.astype(np.uint8, copy=False)
            out["g"] = gg.astype(np.uint8, copy=False)
            out["b"] = bb.astype(np.uint8, copy=False)
            out["a"] = 255

            # 枝の代表点も先頭N点で均等になるよう順序をシャッフル
            if n > 1:
                rng.shuffle(out)

            out.tofile(fp)

            mm._mmap.close()
            del mm
