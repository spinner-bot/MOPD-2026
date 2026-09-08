"""
数据版本管理

按实验方案 §80 定义版本命名和变更记录：

PI-VQA-v0.1 → generator test
PI-VQA-v0.2 → Stage 0
PI-VQA-v0.3 → Pilot
PI-VQA-v1.0 → Formal frozen dataset

v1.0 一旦进入 main experiment:
    Diagnostic test 不再修改

后续修复形成 v1.1 并明确记录差异。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DataVersion:
    """数据版本"""
    version: str  # e.g., "v0.1", "v1.0"
    description: str
    created_at: str  # ISO format
    n_samples: int
    n_images: int

    # 统计
    prp_distribution: dict = field(default_factory=dict)  # {"P": n, "R": n, "PR": n}
    depth_distribution: dict = field(default_factory=dict)  # {"D1": n, "D2": n, ...}
    strategy_distribution: dict = field(default_factory=dict)

    # 状态
    is_frozen: bool = False
    parent_version: Optional[str] = None
    changelog: list[str] = field(default_factory=list)


# === 预定义版本 ===

VERSION_REGISTRY = {
    "v0.1": DataVersion(
        version="v0.1",
        description="Generator test - initial data generation pipeline testing",
        created_at="",  # 运行时填充
        n_samples=100,
        n_images=100,
        is_frozen=False
    ),
    "v0.2": DataVersion(
        version="v0.2",
        description="Stage 0 screening dataset",
        created_at="",
        n_samples=500,
        n_images=450,  # 按 image 去重后
        is_frozen=False,
        parent_version="v0.1"
    ),
    "v0.3": DataVersion(
        version="v0.3",
        description="Pilot dataset for Stage 1 T-only training",
        created_at="",
        n_samples=2000,
        n_images=1800,
        is_frozen=False,
        parent_version="v0.2"
    ),
    "v1.0": DataVersion(
        version="v1.0",
        description="Formal frozen dataset for main experiment",
        created_at="",
        n_samples=11000,  # 8K train + 1K val + 2K test
        n_images=10000,
        is_frozen=True,  # v1.0 必须冻结
        parent_version="v0.3"
    ),
}


def get_version(version: str) -> DataVersion:
    """获取版本信息"""
    return VERSION_REGISTRY.get(version)


def create_new_version(
    version: str,
    description: str,
    n_samples: int,
    n_images: int,
    parent_version: Optional[str] = None,
    changelog: list[str] = None
) -> DataVersion:
    """
    创建新版本

    Args:
        version: 版本号（如 "v1.1"）
        description: 版本描述
        n_samples: 样本数
        n_images: 图像数
        parent_version: 父版本
        changelog: 变更日志

    Returns:
        DataVersion
    """
    new_version = DataVersion(
        version=version,
        description=description,
        created_at=datetime.now().isoformat(),
        n_samples=n_samples,
        n_images=n_images,
        parent_version=parent_version,
        changelog=changelog or []
    )

    # 注册
    VERSION_REGISTRY[version] = new_version
    return new_version


def freeze_version(version: str):
    """
    冻结版本

    冻结后 diagnostic test 不再修改
    """
    if version in VERSION_REGISTRY:
        VERSION_REGISTRY[version].is_frozen = True
    else:
        raise ValueError(f"Version {version} not found")


def validate_version_chain() -> list[str]:
    """
    验证版本链完整性

    Returns:
        错误列表
    """
    errors = []

    for version, data in VERSION_REGISTRY.items():
        if data.parent_version:
            if data.parent_version not in VERSION_REGISTRY:
                errors.append(
                    f"Version {version} has missing parent: {data.parent_version}"
                )
            else:
                # 检查样本数是否递增
                parent = VERSION_REGISTRY[data.parent_version]
                if data.n_samples < parent.n_samples:
                    errors.append(
                        f"Version {version} has fewer samples than parent {data.parent_version}"
                    )

    return errors


def get_version_path(version: str) -> list[str]:
    """
    获取版本路径（从初始到当前）

    Returns:
        [v0.1, v0.2, ..., version]
    """
    path = [version]
    current = version

    while current in VERSION_REGISTRY:
        parent = VERSION_REGISTRY[current].parent_version
        if parent is None:
            break
        path.append(parent)
        current = parent

    return list(reversed(path))


def compute_dataset_hash(data_path: str) -> str:
    """
    计算数据集 hash

    用于追踪数据版本（方案 §79: dataset_hash）
    """
    import hashlib
    import os

    hasher = hashlib.sha256()

    if os.path.isfile(data_path):
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    elif os.path.isdir(data_path):
        # 对目录，按文件名排序后计算
        for root, dirs, files in os.walk(data_path):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(root, fname)
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)

    return hasher.hexdigest()[:16]  # 截短


@dataclass
class DatasetManifest:
    """数据集清单（方案 §79, §80）"""
    version: str
    hash: str
    created_at: str
    n_samples: int
    n_images: int
    splits: dict  # {"train": n, "val": n, "test": n, "stage0": n}
    prp_distribution: dict
    depth_distribution: dict
    is_frozen: bool
    file_paths: dict  # {"train": path, ...}

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "hash": self.hash,
            "created_at": self.created_at,
            "n_samples": self.n_samples,
            "n_images": self.n_images,
            "splits": self.splits,
            "prp_distribution": self.prp_distribution,
            "depth_distribution": self.depth_distribution,
            "is_frozen": self.is_frozen,
            "file_paths": self.file_paths
        }
