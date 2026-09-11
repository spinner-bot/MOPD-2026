#!/usr/bin/env python3
"""
Prepare probe batch from Vision-OPD-6K dataset
"""

import json
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_probe_batch(
    dataset_name="yuanqianhao/Vision-OPD-6K",
    output_path="./probe_data/probe_batch.jsonl",
    num_samples=100,
    model_path="/root/hjc/models/Qwen2.5-0.5B-Instruct",
    seed=42
):
    """
    Prepare probe batch from Vision-OPD-6K dataset

    Args:
        dataset_name: HuggingFace dataset name
        output_path: Output JSONL file path
        num_samples: Number of samples to extract
        model_path: Path to tokenizer model
        seed: Random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    print(f"Dataset size: {len(dataset)}")

    # Randomly sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    print(f"Selected {len(indices)} samples for probe batch")

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process each sample
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]

            # Extract prompt and tokenize
            prompt = sample.get('prompt', '')
            if not prompt and 'conversations' in sample:
                # Extract from conversations format
                for conv in sample['conversations']:
                    if conv.get('role') == 'user' or conv.get('from') == 'human':
                        prompt = conv.get('content', conv.get('value', ''))
                        break

            if not prompt:
                print(f"Warning: Sample {sample_idx} has no prompt, skipping")
                continue

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )

            probe_sample = {
                'id': sample_idx,
                'input_ids': inputs['input_ids'].squeeze(0).tolist(),
                'attention_mask': inputs['attention_mask'].squeeze(0).tolist(),
                'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
                'original_index': sample_idx
            }

            f.write(json.dumps(probe_sample, ensure_ascii=False) + '\n')

            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{len(indices)} samples")

    print(f"\nProbe batch saved to {output_path}")
    print(f"Total samples: {len(indices)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare probe batch from Vision-OPD-6K")
    parser.add_argument("--dataset", type=str, default="yuanqianhao/Vision-OPD-6K")
    parser.add_argument("--output", type=str, default="./probe_data/probe_batch.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--model-path", type=str, default="/root/hjc/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    prepare_probe_batch(
        dataset_name=args.dataset,
        output_path=args.output,
        num_samples=args.num_samples,
        model_path=args.model_path,
        seed=args.seed
    )
