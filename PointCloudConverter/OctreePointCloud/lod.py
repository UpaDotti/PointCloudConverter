import json
import struct
import numpy as np
from tqdm import tqdm

from PointCloudConverter import utils
from .config import Config
from .octree import dict_to_key


def build_leaf_lods(config: Config) -> None:
    in_dtype = config.options.in_dtype
    lod_steps = config.options.lod_steps
    nodes_dir = config.paths.nodes_dir
    leaf_lods_path = config.paths.leaf_lods_path
    oct_manifest_path = config.paths.oct_manifest_path

    with oct_manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    leaves = manifest["leaves"]
    leaves_sorted = sorted(
        leaves, key=lambda x: (int(x["key"]["depth"]), int(x["key"]["path"]))
    )

    offset = 0
    point_struct = struct.Struct("<HHHBBBB")
    leaf_records = []

    for leaf in tqdm(leaves_sorted, desc="LOD構築", unit="leaf"):
        key = dict_to_key(leaf["key"])
        leaf_bin = nodes_dir / leaf["file"]

        mm = np.memmap(leaf_bin, mode="r", dtype=in_dtype)
        n = int(mm.shape[0])

        used = np.zeros(n, bool)
        cum: dict[int, int] = {}

        for s in sorted(lod_steps, reverse=True):
            idx = utils.downsample_indices(n, step=int(s))
            new = idx[~used[idx]]
            used[new] = True
            cum[int(s)] = int(used.sum())

        total_points = cum[int(lod_steps[0])]
        base = offset
        lods = [(base, cum[int(s)]) for s in lod_steps]
        offset += total_points * point_struct.size

        leaf_records.append(
            {
                "key": {"depth": key.depth, "path": key.path},
                "min": leaf["min"],
                "max": leaf["max"],
                "count": int(leaf["count"]),
                "file": leaf["file"],
                "lods": [[int(a), int(b)] for (a, b) in lods],
            }
        )

        mm._mmap.close()
        del mm

    out = {
        "lod_steps": [int(s) for s in lod_steps],
        "point_struct": "<HHHBBBB",
        "leaf_count": len(leaf_records),
        "total_body_bytes": int(offset),
        "leaf_records": leaf_records,
    }

    with leaf_lods_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
