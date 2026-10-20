import json
import struct
import numpy as np
from tqdm import tqdm

from PointCloudConverter import utils
from .config import Config
from .octree import BRANCHES_FILENAME, LEAVES_FILENAME


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
    leaves_bin = nodes_dir / LEAVES_FILENAME
    branches_bin = nodes_dir / BRANCHES_FILENAME
    with opc_path.open("ab") as fp:

        # leafループ（葉の点はオクトツリー構築時に量子化済み）
        with leaves_bin.open("rb") as src:
            for record in tqdm(leaf_records, desc="OPCボディ出力(Leaf)", unit="leaf"):

                n = int(record["point_count"])
                src.seek(int(record["offset"]) * out_dtype.itemsize)
                out = np.fromfile(src, dtype=out_dtype, count=n)

                # 描画時の動的間引きで偏りが出ないよう順序をシャッフル
                if n > 1:
                    rng.shuffle(out)

                out.tofile(fp)

        # 書き出し済みの葉binは以降使わないため削除（ピーク使用量を削減）
        leaves_bin.unlink(missing_ok=True)

        # branchループ
        with branches_bin.open("rb") as src:
            for record in tqdm(branch_records, desc="OPCボディ出力(Branch)", unit="branch"):

                # 代表点が無い枝はスキップ
                n = int(record["point_count"])
                if n == 0:
                    continue

                # 代表点を読み込む
                src.seek(int(record["offset"]) * in_dtype.itemsize)
                pts = np.fromfile(src, dtype=in_dtype, count=n)

                # 量子化用のboundsで出力配列を構成
                out = utils.quantize_points(pts, record["min"], record["max"], out_dtype)

                # 枝の代表点も先頭N点で均等になるよう順序をシャッフル
                if n > 1:
                    rng.shuffle(out)

                out.tofile(fp)
