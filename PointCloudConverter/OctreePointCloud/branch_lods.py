import json
import numpy as np
from tqdm import tqdm

from .config import Config
from .octree import BRANCHES_FILENAME, array_bounds, dict_to_key, make_child_key, sample_for_parent


def build_branch_lods(config: Config) -> None:
    """枝ごとの代表点を生成（ボトムアップで間引き）"""

    in_dtype = config.options.in_dtype
    max_per_branch = config.options.max_points_per_branch
    nodes_dir = config.paths.nodes_dir
    oct_manifest_path = config.paths.oct_manifest_path
    leaf_lods_path = config.paths.leaf_lods_path
    branch_lods_path = config.paths.branch_lods_path

    # オクトツリーマニフェストを読込
    with oct_manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    branches = manifest["branches"]
    leaves = manifest["leaves"]

    # leafメタを読込（base_offsetの開始位置を取るため）
    with leaf_lods_path.open("r", encoding="utf-8") as f:
        leaf_lod_data = json.load(f)
    leaf_total_points = int(leaf_lod_data["total_points"])

    # path → 葉の間引き済みの点の位置（ファイル上で分割した枝の子になった葉だけが持つ）
    leaf_sample_by_path: dict[int, dict] = {}
    for leaf in leaves:
        if "sample" in leaf:
            leaf_sample_by_path[int(leaf["key"]["path"])] = leaf["sample"]

    # 深い枝から処理（ボトムアップ）
    branches_sorted = sorted(branches, key=lambda b: -int(b["key"]["depth"]))

    # path → 代表点データ（point_count / offset / min / max）
    branch_data_by_path: dict[int, dict] = {}

    # 1つの子から取る最大点数
    max_per_child = max(1, max_per_branch // 8)

    # 枝の代表点ファイル（子の代表点を読み、生成した代表点を末尾に追記する）
    with (nodes_dir / BRANCHES_FILENAME).open("r+b") as fp:

        # 枝ループ
        for branch in tqdm(branches_sorted, desc="Branch代表点生成", unit="branch"):

            key = dict_to_key(branch["key"])

            # オクトツリー構築時（メモリ上の部分木）に生成済みの代表点はそのまま使う
            if "lod" in branch:
                branch_data_by_path[key.path] = branch["lod"]
                continue

            # 子から代表点を集める
            chunks = []
            for child_idx in range(8):
                if not branch["children"][child_idx]:
                    continue

                child_key = make_child_key(key, child_idx)
                child_path = child_key.path

                # 子のソースを選ぶ（枝なら代表点 / 葉なら間引き済みの点）
                if child_path in branch_data_by_path:
                    src = branch_data_by_path[child_path]
                elif child_path in leaf_sample_by_path:
                    src = leaf_sample_by_path[child_path]
                else:
                    continue

                # ソースを読み込む（代表点 / 間引き済みの点なので小さい）
                n = int(src["point_count"])
                if n == 0:
                    continue
                fp.seek(int(src["offset"]) * in_dtype.itemsize)
                pts = np.fromfile(fp, dtype=in_dtype, count=n)

                # 固定間隔で間引いて上限まで取る
                chunks.append(sample_for_parent(pts, max_per_child))

            # 子が一つもなければスキップ
            if not chunks:
                continue

            # 連結して末尾に追記
            summary = np.concatenate(chunks)
            fp.seek(0, 2)
            offset = fp.tell() // in_dtype.itemsize
            summary.tofile(fp)

            # 保存boundsは幾何セルではなく、代表点の実範囲(tight)にする（葉と同じ方針）
            bounds_min, bounds_max = array_bounds(summary)

            # 代表点データを保存
            branch_data_by_path[key.path] = {
                "point_count": int(summary.size),
                "offset": offset,
                "min": bounds_min,
                "max": bounds_max,
            }

    # depth → path 順に並べてbranch_recordsを構築
    branches_in_order = sorted(
        branches, key=lambda b: (int(b["key"]["depth"]), int(b["key"]["path"]))
    )

    branch_records = []
    base_offset = leaf_total_points
    for branch in branches_in_order:
        path = int(branch["key"]["path"])
        data = branch_data_by_path[path]
        record = {
            "key": branch["key"],
            "min": data["min"],
            "max": data["max"],
            "point_count": data["point_count"],
            "offset": data["offset"],
            "base_offset": base_offset,
        }
        base_offset += data["point_count"]
        branch_records.append(record)

    # メタ全体を構築
    out = {
        "branch_count": len(branch_records),
        "branch_records": branch_records,
    }

    # メタを出力
    with branch_lods_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
