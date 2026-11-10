import argparse
from pathlib import Path

from .config import Paths, Options, Config
from .points_writer import write_child_bins
from .octree import build_octree
from .leaf_lods import build_leaf_lods
from .branch_lods import build_branch_lods
from .opc_writer import write_opc
from .cleanup import cleanup_intermediate_files


def parse_args() -> argparse.Namespace:
    """コマンドライン引数（入力フォルダと出力フォルダ）を読む"""
    parser = argparse.ArgumentParser(description=".las（フォルダ）から.opcを出力する")
    parser.add_argument("las_dir", type=Path, help="入力の.lasファイルがあるフォルダ")
    parser.add_argument("out_dir", type=Path, help="sample.opcと中間ファイルを書き出すフォルダ")
    return parser.parse_args()


def main() -> None:
    """
    .las（フォルダ）から.opcを出力

    フロー：
    1. 点群を8分割キャッシュへ振り分け
    2. オクトツリー構築
    3. LOD構築
    4. 枝の代表点を生成
    5. OPC書き出し（葉binは葉の書き出し後に削除）
    6. キャッシュファイル削除
    """
    args = parse_args()

    # 入力フォルダが無ければ変換を始めずに終える
    if not args.las_dir.is_dir():
        raise SystemExit(f"入力フォルダが見つかりません: {args.las_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = Paths(las_dir=args.las_dir, out_dir=args.out_dir)
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
