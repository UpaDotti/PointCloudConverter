import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from PointCloudConverter import utils
from .config import Config
from .octree import branch_filename, dict_to_key, make_child_key


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

    # path → 葉のbinパス辞書
    leaf_bin_by_path: dict[int, Path] = {}
    for leaf in leaves:
        path = int(leaf["key"]["path"])
        leaf_bin_by_path[path] = nodes_dir / leaf["file"]

    # path → 枝の代表点binパス辞書（処理しながら追加）
    branch_bin_by_path: dict[int, Path] = {}

    # 深い枝から処理（ボトムアップ）
    branches_sorted = sorted(branches, key=lambda b: -int(b["key"]["depth"]))

    # path → 代表点データ（point_count / file）
    branch_data_by_path: dict[int, dict] = {}

    # 1つの子から取る最大点数
    max_per_child = max(1, max_per_branch // 8)

    # 枝ループ
    for branch in tqdm(branches_sorted, desc="Branch代表点生成", unit="branch"):

        key = dict_to_key(branch["key"])
        out_bin = nodes_dir / branch_filename(key)

        # 子から代表点を集める
        chunks = []
        for child_idx in range(8):
            if not branch["children"][child_idx]:
                continue

            child_key = make_child_key(key, child_idx)
            child_path = child_key.path

            # 子のソースbinを選ぶ（枝なら代表点 / 葉なら生点）
            if child_path in branch_bin_by_path:
                src = branch_bin_by_path[child_path]
            elif child_path in leaf_bin_by_path:
                src = leaf_bin_by_path[child_path]
            else:
                continue

            # ソースの点数を取得
            mm = np.memmap(src, mode="r", dtype=in_dtype)
            n = int(mm.shape[0])
            if n == 0:
                mm._mmap.close()
                del mm
                continue

            # 固定間隔で間引いて上限まで取る
            step = max(1, n // max_per_child)
            idx = utils.downsample_indices(n, step=step)
            if idx.size > max_per_child:
                idx = idx[:max_per_child]
            sample = np.array(mm[idx])
            chunks.append(sample)

            mm._mmap.close()
            del mm

        # 子が一つもなければスキップ
        if not chunks:
            continue

        # 連結して書き出し
        summary = np.concatenate(chunks)
        summary.tofile(out_bin)

        # 辞書に登録
        branch_bin_by_path[key.path] = out_bin

        # 保存boundsは幾何セルではなく、代表点の実範囲(tight)にする（葉と同じ方針）
        bounds_min = [
            float(summary["x"].min()),
            float(summary["y"].min()),
            float(summary["z"].min()),
        ]
        bounds_max = [
            float(summary["x"].max()),
            float(summary["y"].max()),
            float(summary["z"].max()),
        ]

        # 代表点データを保存
        branch_data_by_path[key.path] = {
            "point_count": int(summary.size),
            "file": out_bin.name,
            "min": bounds_min,
            "max": bounds_max,
        }

    # depth → path 順に並べてbranch_recordsを構築
    branches_in_order = sorted(
        branches, key=lambda b: (int(b["key"]["depth"]), int(b["key"]["path"]))
    )

    branch_records = []
    offset = leaf_total_points
    for branch in branches_in_order:
        path = int(branch["key"]["path"])
        data = branch_data_by_path[path]
        record = {
            "key": branch["key"],
            "min": data["min"],
            "max": data["max"],
            "point_count": data["point_count"],
            "file": data["file"],
            "base_offset": offset,
        }
        offset += data["point_count"]
        branch_records.append(record)

    # メタ全体を構築
    out = {
        "branch_count": len(branch_records),
        "branch_records": branch_records,
    }

    # メタを出力
    with branch_lods_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
