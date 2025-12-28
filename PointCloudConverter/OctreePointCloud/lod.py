import json
import struct
import numpy as np
from tqdm import tqdm

from PointCloudConverter import utils
from .config import Config
from .octree import dict_to_key


def build_leaf_lods(config: Config) -> None:
    """LODを構築"""

    in_dtype = config.options.in_dtype
    lod_steps = config.options.lod_steps
    nodes_dir = config.paths.nodes_dir
    leaf_lods_path = config.paths.leaf_lods_path
    oct_manifest_path = config.paths.oct_manifest_path

    # オクトツリーもマニュアルを読み込む
    with oct_manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # depth → path の順に昇順ソート
    leaves = manifest["leaves"]
    leaves_sorted = sorted(
        leaves, key=lambda x: (int(x["key"]["depth"]), int(x["key"]["path"]))
    )

    offset = 0
    point_struct = struct.Struct("<HHHBBBB")
    leaf_records = []

    # leafごとのループ / プログレスバー
    for leaf in tqdm(leaves_sorted, desc="LOD構築", unit="leaf"):

        # キーとleafファイルパスを取得
        key = dict_to_key(leaf["key"])
        leaf_bin = nodes_dir / leaf["file"]

        # メモリマップでバイナリを読み込む
        mm = np.memmap(leaf_bin, mode="r", dtype=in_dtype)
        n = int(mm.shape[0])

        used = np.zeros(n, bool)
        cumulative: dict[int, int] = {}

        # LODが粗い順に実行
        for step in sorted(lod_steps, reverse=True):

            # step間隔のインデックス配列
            idx = utils.downsample_indices(n, step=step)

            # step間隔の対象 + まだ使われていない点を取得
            new = idx[~used[idx]]

            # 使用済みフラグを更新
            used[new] = True

            # 今回のStepまでで使用した点数
            cumulative[int(step)] = int(used.sum())

        # このleafの点数 ※LODに1がない場合があるため
        total_points = cumulative[lod_steps[0]]

        # このleafの先頭がどのバイト位置から始まるか
        base = offset

        # LODごとに[このleafの先頭位置,そのLODまでに使用するポイント数]
        lods = [(base, cumulative[s]) for s in lod_steps]

        # このleafのポイント分のバイト数を進める
        offset += total_points * point_struct.size

        # leafデータを追加
        leaf_records.append(
            {
                "key": {"depth": key.depth, "path": key.path},
                "min": leaf["min"],
                "max": leaf["max"],
                "count": leaf["count"],
                "file": leaf["file"],
                "lods": [[a, b] for (a, b) in lods],
            }
        )

        # memmapを閉じる
        mm._mmap.close()
        del mm

    # LOD全体のデータを構築
    out = {
        "lod_steps": [s for s in lod_steps],
        "point_struct": "<HHHBBBB",
        "leaf_count": len(leaf_records),
        "total_body_bytes": offset,
        "leaf_records": leaf_records,
    }

    # LODのデータを出力
    with leaf_lods_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
