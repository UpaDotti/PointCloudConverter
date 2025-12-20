from pathlib import Path

from .config import Paths, Options, Config
from .las_io import compute_global_center, export_las_to_points_bin
from .octree import build_octree
from .lod import build_leaf_lods
from .opc_writer import write_opc
from .cleanup import cleanup_intermediate_files


def main() -> None:
    """
    .las（フォルダ）から.opcを出力

    フロー：
    1. センター計算
    2. 点群のキャッシュファイル作成
    3. オクトツリー構築
    4. LOD構築
    5. OPC書き出し
    6. キャッシュファイル削除
    """
    LAS_DIR = Path(
        r"C:\Users\user\Github\PointCloudViewer\Assets\PointCloudViewer\Las"
    )
    OUT_DIR = Path(r"C:\Users\user\Desktop")

    paths = Paths(las_dir=LAS_DIR, out_dir=OUT_DIR)
    options = Options()
    config = Config(paths=paths, options=options)

    global_center = compute_global_center(config)
    export_las_to_points_bin(config, global_center)
    build_octree(config)
    build_leaf_lods(config)
    write_opc(config)
    cleanup_intermediate_files(config)


if __name__ == "__main__":
    main()
