import json
import numpy as np
from tqdm import tqdm

from .config import Config
from .octree import dict_to_key


def build_leaf_lods(config: Config) -> None:
    """葉ごとの点数とボディ内オフセットをまとめる"""

    in_dtype = config.options.in_dtype
    nodes_dir = config.paths.nodes_dir
    leaf_lods_path = config.paths.leaf_lods_path
    oct_manifest_path = config.paths.oct_manifest_path

    # オクトツリーのマニフェストを読み込む
    with oct_manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # depth → path の順に昇順ソート
    leaves = manifest["leaves"]
    leaves_sorted = sorted(
        leaves, key=lambda x: (int(x["key"]["depth"]), int(x["key"]["path"]))
    )

    offset = 0
    leaf_records = []

    # leafごとのループ / プログレスバー
    for leaf in tqdm(leaves_sorted, desc="Leafメタ構築", unit="leaf"):

        # キーとleafファイルパスを取得
        key = dict_to_key(leaf["key"])
        leaf_bin = nodes_dir / leaf["file"]

        # メモリマップでバイナリを読み込む
        mm = np.memmap(leaf_bin, mode="r", dtype=in_dtype)

        # この葉の点数
        point_count = int(mm.shape[0])

        # 保存boundsは幾何セルではなく「実際の点の範囲(tight)」にする
        # 八分木の分割は幾何セルのまま。ビューアの点サイズ/量子化精度を正しくするため保存値だけ実範囲にする
        if point_count > 0:
            bounds_min = [
                float(mm["x"].min()),
                float(mm["y"].min()),
                float(mm["z"].min()),
            ]
            bounds_max = [
                float(mm["x"].max()),
                float(mm["y"].max()),
                float(mm["z"].max()),
            ]
        else:
            bounds_min = leaf["min"]
            bounds_max = leaf["max"]

        # このleafの先頭がどのバイト位置から始まるか
        base_offset = offset

        # このleafのポイント分の位置を進める
        offset += point_count

        # leafデータを追加
        leaf_records.append(
            {
                "key": {"depth": key.depth, "path": key.path},
                "min": bounds_min,
                "max": bounds_max,
                "count": leaf["count"],
                "file": leaf["file"],
                "base_offset": base_offset,
                "point_count": point_count,
            }
        )

        # memmapを閉じる
        mm._mmap.close()
        del mm

    # 全体のデータを構築
    out = {
        "leaf_count": len(leaf_records),
        "total_points": offset,
        "leaf_records": leaf_records,
    }

    # メタを出力
    with leaf_lods_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
