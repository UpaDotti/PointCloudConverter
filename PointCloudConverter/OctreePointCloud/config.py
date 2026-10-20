from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Options:
    """設定値"""

    # ファイル識別
    magic = b"OPC\x00"
    version_major = 1
    version_minor = 0
    version_patch = 0

    # オクトツリー
    max_depth: int = 8  # 深さ
    max_points_per_leaf: int = 200_000  # 葉ノードの最大点数
    max_points_per_branch: int = 30_000  # 枝ノードの最大代表点数

    # I/O
    pattern: str = "*.las"  # 入力ファイルの検索パターン
    read_size: int = 2_000_000  # 1回で読む点数
    max_points_in_memory: int = 20_000_000  # この点数以下のノードはRAM上で一括分割する（1点18byte＋並べ替え用の作業領域）

    # 中間形式
    in_dtype = np.dtype(
        [
            ("x", "<f4"),  # 32bit
            ("y", "<f4"),
            ("z", "<f4"),
            ("r", "<u2"),  # 16bit
            ("g", "<u2"),
            ("b", "<u2"),
        ],
        align=False,
    )

    # 量子化後の形式
    out_dtype = np.dtype(
        [
            ("x", "<u2"),  # 16bit
            ("y", "<u2"),
            ("z", "<u2"),
            ("r", "u1"),  # 8bit
            ("g", "u1"),
            ("b", "u1"),
        ],
        align=False,
    )


@dataclass(frozen=True)
class Paths:
    """パス"""

    # 入力フォルダと出力フォルダ
    las_dir: Path
    out_dir: Path

    # OPCファイル
    @property
    def opc_path(self) -> Path:
        return self.out_dir / "sample.opc"

    # オクトツリーノードのフォルダ
    @property
    def nodes_dir(self) -> Path:
        return self.out_dir / "nodes"

    # オクトツリーのメタファイル
    @property
    def oct_manifest_path(self) -> Path:
        return self.out_dir / "octree_manifest.json"

    # リーフごとのメタファイル
    @property
    def leaf_lods_path(self) -> Path:
        return self.out_dir / "leaf_lods.json"

    # 枝の代表点メタファイル
    @property
    def branch_lods_path(self) -> Path:
        return self.out_dir / "branch_lods.json"


@dataclass(frozen=True)
class Config:
    """設定値 / パス"""

    options: Options
    paths: Paths
