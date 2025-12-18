import json
import struct
import numpy as np
from tqdm import tqdm

from PointCloudConverter import utils
from .config import Config


def write_opc(config: Config) -> None:
    magic = config.options.magic
    version_major = config.options.version_major
    version_minor = config.options.version_minor
    version_patch = config.options.version_patch
    lod_steps = config.options.lod_steps
    point_dtype = config.options.point_dtype
    oct_manifest_path = config.paths.oct_manifest_path
    leaf_lods_path = config.paths.leaf_lods_path
    nodes_dir = config.paths.nodes_dir
    opc_path = config.paths.opc_path

    with oct_manifest_path.open("r", encoding="utf-8") as f:
        oct_manifest = json.load(f)
    with leaf_lods_path.open("r", encoding="utf-8") as f:
        lod_data = json.load(f)

    root_min = np.asarray(oct_manifest["root_min"], np.float32)
    root_max = np.asarray(oct_manifest["root_max"], np.float32)

    leaf_records = lod_data["leaf_records"]
    point_struct = struct.Struct("<HHHBBBB")

    # --- ヘッダー ---
    with opc_path.open("wb") as fp:
        fp.write(magic)
        fp.write(struct.pack("<B", version_major))
        fp.write(struct.pack("<B", version_minor))
        fp.write(struct.pack("<B", version_patch))

        fp.write(struct.pack("<B", len(lod_steps)))
        fp.write(struct.pack("<3f", *root_min))
        fp.write(struct.pack("<3f", *root_max))
        fp.write(struct.pack("<I", len(leaf_records)))

        for r in leaf_records:
            depth = int(r["key"]["depth"])
            path = int(r["key"]["path"])
            fp.write(struct.pack("<B", depth))
            fp.write(struct.pack("<Q", path))
            fp.write(struct.pack("<3f", *np.asarray(r["min"], np.float32)))
            fp.write(struct.pack("<3f", *np.asarray(r["max"], np.float32)))
            for off, cnt in r["lods"]:
                fp.write(struct.pack("<Q", int(off)))
                fp.write(struct.pack("<I", int(cnt)))

    # --- ボディ ---
    with opc_path.open("ab") as fp:
        for r in tqdm(leaf_records, desc="OPCボディ出力", unit="leaf"):
            leaf_bin = nodes_dir / r["file"]
            mm = np.memmap(leaf_bin, mode="r", dtype=point_dtype)
            n = int(mm.shape[0])

            used = np.zeros(n, bool)
            write_parts = []
            for s in sorted(lod_steps, reverse=True):
                idx = utils.downsample_indices(n, step=int(s))
                new = idx[~used[idx]]
                used[new] = True
                write_parts.append(new)
            order = (
                np.concatenate(write_parts) if write_parts else np.array([], np.int64)
            )

            vmin = np.asarray(r["min"], np.float32)
            vmax = np.asarray(r["max"], np.float32)

            qx = utils.quantize_u16(mm["x"][order], vmin[0], vmax[0])
            qy = utils.quantize_u16(mm["y"][order], vmin[1], vmax[1])
            qz = utils.quantize_u16(mm["z"][order], vmin[2], vmax[2])

            rr = utils.convert_color(mm["r"][order])
            gg = utils.convert_color(mm["g"][order])
            bb = utils.convert_color(mm["b"][order])
            aa = np.full_like(rr, 255, dtype=np.uint8)

            for x16, y16, z16, r8, g8, b8, a8 in zip(qx, qy, qz, rr, gg, bb, aa):
                fp.write(
                    point_struct.pack(
                        int(x16), int(y16), int(z16), int(r8), int(g8), int(b8), int(a8)
                    )
                )

            mm._mmap.close()
            del mm
