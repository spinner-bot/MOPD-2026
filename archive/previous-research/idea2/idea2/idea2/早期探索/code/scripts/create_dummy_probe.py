#!/usr/bin/env python3
"""
创建 dummy probe batch 用于测试
"""

import json
import os
import random


def create_dummy_probe_batch(output_path: str = "./probe_data/probe_batch.jsonl", num_samples: int = 20):
    """
    创建 dummy probe batch

    Args:
        output_path: 输出 JSONL 文件路径
        num_samples: 样本数量
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建 dummy 数据（模拟 Qwen tokenizer 的输出）
    # 假设 seq_len 在 50-200 之间
    vocab_size = 151936  # Qwen3.5 vocab size

    with open(output_path, 'w') as f:
        for i in range(num_samples):
            seq_len = random.randint(50, 200)
            input_ids = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
            attention_mask = [1] * seq_len

            sample = {
                'id': i,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'prompt': f'Dummy prompt {i}',
            }
            f.write(json.dumps(sample) + '\n')

    print(f"Created {num_samples} dummy samples at {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./probe_data/probe_batch.jsonl")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    create_dummy_probe_batch(output_path=args.output, num_samples=args.num_samples)
