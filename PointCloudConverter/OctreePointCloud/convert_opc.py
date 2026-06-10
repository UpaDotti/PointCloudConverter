from pathlib import Path

from .config import Paths, Options, Config
from .points_writer import write_child_bins
from .octree import build_octree
from .leaf_lods import build_leaf_lods
from .branch_lods import build_branch_lods
from .opc_writer import write_opc
from .cleanup import cleanup_intermediate_files


def main() -> None:
    """
    .las（フォルダ）から.opcを出力

    フロー：
    1. 点群を8分割キャッシュへ振り分け
    2. オクトツリー構築
    3. LOD構築
    4. 枝の代表点を生成
    5. OPC書き出し（葉binは書き出し後に逐次削除）
    6. キャッシュファイル削除
    """
    LAS_DIR = Path(r"C:\Users\user\Desktop\点群")
    OUT_DIR = Path(r"C:\Users\user\Desktop")

    paths = Paths(las_dir=LAS_DIR, out_dir=OUT_DIR)
    options = Options()
    config = Config(paths=paths, options=options)

    points_min, points_max, child_counts = write_child_bins(config)
    build_octree(config, points_min, points_max, child_counts)
    build_leaf_lods(config)
    build_branch_lods(config)
    write_opc(config)
    cleanup_intermediate_files(config)


if __name__ == "__main__":
    main()
