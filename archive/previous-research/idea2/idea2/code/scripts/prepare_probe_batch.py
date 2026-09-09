#!/usr/bin/env python3
"""
准备 probe batch 数据：从 Vision-OPD-6K 数据集中抽取 100 条样本
"""

import json
import os
import random
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_probe_batch(
    dataset_name: str = "yuanqianhao/Vision-OPD-6K",
    output_path: str = "./probe_data/probe_batch.jsonl",
    num_samples: int = 100,
    model_name: str = "Qwen/Qwen3.5-4B",
    seed: int = 42,
):
    """
    准备 probe batch

    Args:
        dataset_name: HuggingFace 数据集名称
        output_path: 输出 JSONL 文件路径
        num_samples: 抽取样本数量
        model_name: tokenizer 模型名称
        seed: 随机种子
    """
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading dataset from {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    print(f"Dataset size: {len(dataset)}")

    # 随机抽取样本
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    print(f"Sampling {len(indices)} samples...")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 处理每个样本
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]

            # 提取 prompt 和 image（如果有）
            # 注意：这里假设数据集有 'prompt' 或 'conversations' 字段
            if 'prompt' in sample:
                prompt = sample['prompt']
            elif 'conversations' in sample and len(sample['conversations']) > 0:
                # 提取用户的第一条消息
                for conv in sample['conversations']:
                    if conv.get('role') == 'user' or conv.get('from') == 'human':
                        prompt = conv.get('content', conv.get('value', ''))
                        break
                else:
                    prompt = sample['conversations'][0].get('content', sample['conversations'][0].get('value', ''))
            else:
                print(f"WARN: sample {sample_idx} has no prompt, skipping")
                continue

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs['input_ids'].squeeze(0).tolist()
            attention_mask = inputs['attention_mask'].squeeze(0).tolist()

            # 保存
            probe_sample = {
                'id': sample_idx,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,  # 截断保存
            }
            f.write(json.dumps(probe_sample, ensure_ascii=False) + '\n')

            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{len(indices)} samples")

    print(f"Saved {len(indices)} samples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare probe batch from Vision-OPD-6K")
    parser.add_argument("--dataset", type=str, default="yuanqianhao/Vision-OPD-6K")
    parser.add_argument("--output", type=str, default="./probe_data/probe_batch.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    prepare_probe_batch(
        dataset_name=args.dataset,
        output_path=args.output,
        num_samples=args.num_samples,
        model_name=args.model,
        seed=args.seed,
    )
