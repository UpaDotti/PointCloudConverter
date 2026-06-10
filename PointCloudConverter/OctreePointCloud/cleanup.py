import shutil
from .config import Config


def cleanup_intermediate_files(config: Config) -> None:
    oct_manifest_path = config.paths.oct_manifest_path
    leaf_lods_path = config.paths.leaf_lods_path
    branch_lods_path = config.paths.branch_lods_path
    nodes_dir = config.paths.nodes_dir

    if oct_manifest_path.exists():
        oct_manifest_path.unlink()

    if leaf_lods_path.exists():
        leaf_lods_path.unlink()

    if branch_lods_path.exists():
        branch_lods_path.unlink()

    if nodes_dir.exists() and nodes_dir.is_dir():
        shutil.rmtree(nodes_dir)
