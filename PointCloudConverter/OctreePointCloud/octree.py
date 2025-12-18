from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from .config import Config
from .points_cache import compute_points_aabb, append_points


@dataclass(frozen=True)
class NodeKey:
    depth: int
    path: int


def leaf_filename(key: NodeKey) -> str:
    return f"leaf_d{key.depth}_p{key.path}.bin"


def key_to_dict(k: NodeKey) -> dict:
    return {"depth": int(k.depth), "path": int(k.path)}


def dict_to_key(d: dict) -> NodeKey:
    return NodeKey(depth=int(d["depth"]), path=int(d["path"]))


def child_aabb(vmin, vmax, cidx: int):
    vmin = np.asarray(vmin, np.float32)
    vmax = np.asarray(vmax, np.float32)
    center = (vmin + vmax) * 0.5

    x = (cidx >> 2) & 1
    y = (cidx >> 1) & 1
    z = (cidx >> 0) & 1
    is_upper = np.array([x, y, z], dtype=np.uint8)

    child_min = np.where(is_upper == 0, vmin, center)
    child_max = np.where(is_upper == 0, center, vmax)
    return child_min, child_max


def split_node_file(
    config: Config,
    node_bin: Path,
    parent_key: NodeKey,
    vmin,
    vmax,
) -> dict[int, tuple[Path, np.ndarray, np.ndarray, int]]:
    read_size = config.options.read_size
    point_dtype = config.options.point_dtype
    nodes_dir = config.paths.nodes_dir

    vmin = np.asarray(vmin, np.float32)
    vmax = np.asarray(vmax, np.float32)
    center = (vmin + vmax) * 0.5
    cx, cy, cz = center.tolist()

    mm = np.memmap(node_bin, mode="r", dtype=point_dtype)
    n = int(mm.shape[0])
    if n == 0:
        mm._mmap.close()
        del mm
        node_bin.unlink(missing_ok=True)
        return {}

    child_bins: dict[int, Path] = {}
    child_counts = np.zeros(8, dtype=np.int64)

    for cid in range(8):
        child_key = NodeKey(
            depth=parent_key.depth + 1, path=(parent_key.path << 3) | cid
        )
        child_bin = nodes_dir / f"tmp_d{child_key.depth}_p{child_key.path}.bin"
        child_bin.unlink(missing_ok=True)
        child_bins[cid] = child_bin

    for start in range(0, n, read_size):
        part = mm[start : start + read_size]
        xs, ys, zs = part["x"], part["y"], part["z"]

        cidx = (
            ((xs >= cx).astype(np.uint8) << 2)
            | ((ys >= cy).astype(np.uint8) << 1)
            | ((zs >= cz).astype(np.uint8) << 0)
        )

        for cid in range(8):
            sel = cidx == cid
            if np.any(sel):
                recs = part[sel].copy()
                append_points(child_bins[cid], recs)
                child_counts[cid] += recs.size

        del part

    mm._mmap.close()
    del mm
    node_bin.unlink(missing_ok=True)

    out: dict[int, tuple[Path, np.ndarray, np.ndarray, int]] = {}
    for cid in range(8):
        cnt = int(child_counts[cid])
        if cnt <= 0:
            child_bins[cid].unlink(missing_ok=True)
            continue
        mn, mx = child_aabb(vmin, vmax, cid)
        out[cid] = (child_bins[cid], mn, mx, cnt)

    return out


def build_octree(config: Config) -> None:
    read_size = config.options.read_size
    point_dtype = config.options.point_dtype
    max_depth = config.options.max_depth
    max_points_per_leaf = config.options.max_points_per_leaf
    nodes_dir = config.paths.nodes_dir
    points_path = config.paths.points_path
    oct_manifest_path = config.paths.oct_manifest_path

    nodes_dir.mkdir(parents=True, exist_ok=True)

    root_min, root_max, total = compute_points_aabb(config)

    root_key = NodeKey(depth=0, path=1)
    root_bin = nodes_dir / "root.bin"
    root_bin.unlink(missing_ok=True)

    with points_path.open("rb") as fp, tqdm(
        total=total, desc="root作成", unit="pts"
    ) as pbar:
        processed = 0
        while processed < total:
            n = min(read_size, total - processed)
            recs = np.fromfile(fp, dtype=point_dtype, count=n)
            if recs.size == 0:
                break
            append_points(root_bin, recs)
            processed += int(recs.size)
            pbar.update(int(recs.size))

    queue = {root_key: (root_bin, root_min, root_max, total)}
    branches = []
    leaves = []

    with tqdm(desc="オクトツリー分割", unit="node") as pbar:
        while queue:
            key, (bin_path, vmin, vmax, count) = queue.popitem()
            pbar.update(1)

            should_split = key.depth < max_depth and count > max_points_per_leaf

            if not should_split:
                leaf_bin = nodes_dir / leaf_filename(key)
                leaf_bin.unlink(missing_ok=True)
                bin_path.replace(leaf_bin)
                leaves.append(
                    {
                        "key": key_to_dict(key),
                        "min": np.asarray(vmin, np.float32).tolist(),
                        "max": np.asarray(vmax, np.float32).tolist(),
                        "count": int(count),
                        "file": leaf_bin.name,
                    }
                )
                continue

            split = split_node_file(config, bin_path, key, vmin, vmax)
            child_flags = [False] * 8

            for cid, (child_bin, mn, mx, cn) in split.items():
                child_flags[cid] = True
                child_key = NodeKey(depth=key.depth + 1, path=(key.path << 3) | cid)
                queue[child_key] = (child_bin, mn, mx, cn)

            branches.append(
                {
                    "key": key_to_dict(key),
                    "min": np.asarray(vmin, np.float32).tolist(),
                    "max": np.asarray(vmax, np.float32).tolist(),
                    "count": int(count),
                    "children": child_flags,
                }
            )

    manifest = {
        "root_min": np.asarray(root_min, np.float32).tolist(),
        "root_max": np.asarray(root_max, np.float32).tolist(),
        "max_depth": int(max_depth),
        "max_points_per_leaf": int(max_points_per_leaf),
        "branches": branches,
        "leaves": leaves,
    }

    with oct_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
