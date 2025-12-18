from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import numpy as np

from PointCloudConverter import utils


@dataclass(frozen=True)
class Options:
    magic = b"TPC\x00"
    version_major = 1
    version_minor = 0
    version_patch = 0

    # octree
    max_depth: int = 8
    max_points_per_leaf: int = 200_000

    # lod
    lod_levels: int = 7
    lod_first_step: int = 1
    lod_step_factor: int = 4
    lod_steps: Tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        steps = utils.make_lod_steps(
            self.lod_levels, self.lod_first_step, self.lod_step_factor
        )
        object.__setattr__(self, "lod_steps", tuple(map(int, steps)))

    read_size: int = 2_000_000
    pattern: str = "*.las"

    in_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("r", "<u2"),
            ("g", "<u2"),
            ("b", "<u2"),
        ],
        align=False,
    )

    out_dtype = np.dtype(
        [
            ("x", "<u2"),
            ("y", "<u2"),
            ("z", "<u2"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
            ("a", "u1"),
        ],
        align=False,
    )


@dataclass(frozen=True)
class Paths:
    las_dir: Path
    out_dir: Path

    @property
    def opc_path(self) -> Path:
        return self.out_dir / "sample.opc"

    @property
    def points_path(self) -> Path:
        return self.out_dir / "points.bin"

    @property
    def nodes_dir(self) -> Path:
        return self.out_dir / "nodes"

    @property
    def oct_manifest_path(self) -> Path:
        return self.out_dir / "octree_manifest.json"

    @property
    def leaf_lods_path(self) -> Path:
        return self.out_dir / "leaf_lods.json"


@dataclass(frozen=True)
class Config:
    options: Options
    paths: Paths
