"""
数据集划分工具

按实验方案 §8 实现：
- 以 original_image 为单位划分（严禁同一图像的不同问题分到 train 和 test）
- 输出 Train/Val/Test/Stage0 splits
- 支持 P/R/PR 分层抽样

方案 §8:
    严禁:
        Image A / Question 1 → train
        Image A / Question 2 → test

    正确方式:
        I_i ∈ Train ⇒ ∀ Q(I_i) ∈ Train
"""

import json
import numpy as np
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass
import os


@dataclass
class SplitConfig:
    """数据划分配置（方案 §7）"""
    train_size: int = 8000
    val_size: int = 1000
    test_size: int = 2000
    stage0_size: int = 500  # 400-600

    # P/R/PR 比例目标
    target_prp_ratio: tuple[float, float, float] = (1/3, 1/3, 1/3)

    # 随机种子
    seed: int = 42


def load_samples(jsonl_path: str) -> list[dict]:
    """加载 JSONL 格式的样本"""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def group_by_image(samples: list[dict]) -> dict[str, list[dict]]:
    """
    按 source_image_id 分组

    Returns:
        {image_id: [sample1, sample2, ...]}
    """
    groups = defaultdict(list)
    for sample in samples:
        image_id = sample.get("source_image_id", sample.get("image_original", ""))
        groups[image_id].append(sample)
    return dict(groups)


def stratified_split(
    samples: list[dict],
    config: SplitConfig,
    task_family_key: str = "task_family"
) -> dict[str, list[dict]]:
    """
    分层划分数据集

    按 source_image_id 分组后，再按 task_family (P/R/PR) 分层抽样，
    确保每个 split 中 P/R/PR 比例接近目标。

    Args:
        samples: 样本列表
        config: 划分配置
        task_family_key: 任务类型字段名

    Returns:
        {
            "train": [...],
            "val": [...],
            "test": [...],
            "stage0": [...],
            "stats": {...}
        }
    """
    np.random.seed(config.seed)

    # 按 image 分组
    image_groups = group_by_image(samples)
    image_ids = list(image_groups.keys())

    # 按 task_family 进一步分组
    images_by_family = defaultdict(list)
    for img_id in image_ids:
        group = image_groups[img_id]
        # 使用该组中第一个样本的 task_family
        family = group[0].get(task_family_key, "P")
        images_by_family[family].append(img_id)

    # 对每个 family 分别划分
    splits = {"train": [], "val": [], "test": [], "stage0": []}

    for family, family_image_ids in images_by_family.items():
        np.random.shuffle(family_image_ids)
        n_total = len(family_image_ids)

        # 计算该 family 在各 split 中的目标数量
        # 按比例分配
        train_ratio = config.train_size / (config.train_size + config.val_size + config.test_size)
        val_ratio = config.val_size / (config.train_size + config.val_size + config.test_size)
        test_ratio = config.test_size / (config.train_size + config.val_size + config.test_size)

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)

        # 划分
        train_imgs = family_image_ids[:n_train]
        val_imgs = family_image_ids[n_train:n_train + n_val]
        test_imgs = family_image_ids[n_train + n_val:n_train + n_val + n_test]

        # 剩余放入 test（确保不浪费）
        remaining = family_image_ids[n_train + n_val + n_test:]
        test_imgs.extend(remaining)

        # 将 image 中的样本添加到对应 split
        for img_id in train_imgs:
            splits["train"].extend(image_groups[img_id])
        for img_id in val_imgs:
            splits["val"].extend(image_groups[img_id])
        for img_id in test_imgs:
            splits["test"].extend(image_groups[img_id])

    # Stage 0: 从 test 中抽取 400-600 样本（方案 §19）
    # 确保覆盖 P/R/PR 和 depth 1-4+
    test_samples = splits["test"].copy()
    np.random.shuffle(test_samples)
    splits["stage0"] = test_samples[:config.stage0_size]

    # 统计信息
    stats = {
        "total_samples": len(samples),
        "total_images": len(image_ids),
        "splits": {}
    }

    for split_name, split_samples in splits.items():
        if split_name == "stage0":
            continue
        images_in_split = len(set(s.get("source_image_id", s.get("image_original", ""))
                                   for s in split_samples))
        family_counts = defaultdict(int)
        for s in split_samples:
            family = s.get(task_family_key, "unknown")
            family_counts[family] += 1

        stats["splits"][split_name] = {
            "n_samples": len(split_samples),
            "n_images": images_in_split,
            "family_distribution": dict(family_counts)
        }

    splits["stats"] = stats
    return splits


def validate_split(splits: dict) -> list[str]:
    """
    验证划分正确性（方案 §8）

    检查：
    1. 同一 image 的问题不会被分到不同 split
    2. 没有数据泄漏

    Returns:
        错误列表
    """
    errors = []

    # 收集各 split 的 image IDs
    train_images = set()
    val_images = set()
    test_images = set()

    for sample in splits["train"]:
        img_id = sample.get("source_image_id", sample.get("image_original", ""))
        train_images.add(img_id)

    for sample in splits["val"]:
        img_id = sample.get("source_image_id", sample.get("image_original", ""))
        val_images.add(img_id)

    for sample in splits["test"]:
        img_id = sample.get("source_image_id", sample.get("image_original", ""))
        test_images.add(img_id)

    # 检查交集
    if train_images & val_images:
        errors.append(f"Train-Val image overlap: {len(train_images & val_images)} images")
    if train_images & test_images:
        errors.append(f"Train-Test image overlap: {len(train_images & test_images)} images")
    if val_images & test_images:
        errors.append(f"Val-Test image overlap: {len(val_images & test_images)} images")

    return errors


def save_splits(
    splits: dict,
    output_dir: str,
    prefix: str = "PI-VQA-v0.1"
):
    """
    保存划分结果

    Args:
        splits: split 字典
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name in ["train", "val", "test", "stage0"]:
        if split_name not in splits:
            continue
        output_path = os.path.join(output_dir, f"{prefix}_{split_name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in splits[split_name]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Saved {split_name}: {len(splits[split_name])} samples -> {output_path}")

    # 保存统计信息
    stats_path = os.path.join(output_dir, f"{prefix}_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(splits.get("stats", {}), f, ensure_ascii=False, indent=2)
    print(f"Saved stats -> {stats_path}")


def create_splits_from_vision_opd(
    vision_opd_jsonl: str,
    output_dir: str,
    config: Optional[SplitConfig] = None
):
    """
    从 Vision-OPD-6K 数据集创建划分

    Args:
        vision_opd_jsonl: Vision-OPD 数据路径
        output_dir: 输出目录
        config: 划分配置
    """
    if config is None:
        config = SplitConfig()

    print(f"Loading samples from {vision_opd_jsonl}...")
    samples = load_samples(vision_opd_jsonl)
    print(f"Loaded {len(samples)} samples")

    print("Performing stratified split...")
    splits = stratified_split(samples, config)

    # 验证
    errors = validate_split(splits)
    if errors:
        print("WARNING: Split validation errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("Split validation passed: no image overlap between splits")

    # 打印统计
    print("\n=== Split Statistics ===")
    stats = splits.get("stats", {})
    print(f"Total samples: {stats.get('total_samples', 0)}")
    print(f"Total images: {stats.get('total_images', 0)}")
    for split_name, split_stats in stats.get("splits", {}).items():
        print(f"\n{split_name}:")
        print(f"  Samples: {split_stats['n_samples']}")
        print(f"  Images: {split_stats['n_images']}")
        print(f"  Family distribution: {split_stats['family_distribution']}")

    # 保存
    save_splits(splits, output_dir)
    return splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset splits")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--stage0-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = SplitConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        stage0_size=args.stage0_size,
        seed=args.seed
    )

    create_splits_from_vision_opd(args.input, args.output, config)
