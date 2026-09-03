"""
引导性问题生成脚本

使用 DeepSeek API 为 Vision-OPD 数据集中的每个样本生成引导性问题。

用法：
    python generate_guiding_questions.py \
        --input-file data/raw/train.jsonl \
        --output-file data/guiding_questions/raw_generated/batch_001.json \
        --batch-size 100 \
        --api-key <your-api-key>
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any
from pathlib import Path


# 引导性问题生成 prompt
GENERATION_PROMPT = """你是一个推理引导专家。你的任务是为一位正在解决复杂视觉问题的学生提供引导。

给定：
- 问题：{question}
- 答案：{answer}

请生成 2-4 个引导性问题，帮助学生逐步思考并解决问题。

要求：
1. 循序渐进：从识别关键信息 → 应用相关知识 → 整合推理 → 得出结论
2. 每步聚焦：每个引导问题只关注一个推理步骤
3. 启发性：引导学生思考，而非直接给出答案
4. 简洁性：每个引导问题控制在 20 字以内
5. 不要提及"图像"、"图片"等词汇，专注于问题本身

输出格式（严格 JSON）：
{{
  "guiding_questions": [
    "引导问题1",
    "引导问题2",
    "引导问题3"
  ]
}}

现在请为以下问题生成引导性问题：

问题：{question}
答案：{answer}

请输出 JSON："""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成引导性问题")
    parser.add_argument("--input-file", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output-file", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--batch-size", type=int, default=100, help="批次大小")
    parser.add_argument("--api-key", required=True, help="DeepSeek API key")
    parser.add_argument("--start-idx", type=int, default=0, help="起始索引")
    parser.add_argument("--end-idx", type=int, default=None, help="结束索引")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--retry-delay", type=int, default=5, help="重试延迟（秒）")
    return parser.parse_args()


def call_deepseek_api(prompt: str, api_key: str, max_retries: int = 3, retry_delay: int = 5) -> Dict[str, Any]:
    """调用 DeepSeek API"""
    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers=headers,
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                return {"success": True, "content": content}

        except urllib.error.HTTPError as e:
            error_msg = e.read().decode("utf-8")
            print(f"  HTTP Error {e.code} (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        except Exception as e:
            print(f"  Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return {"success": False, "error": "Max retries exceeded"}


def parse_guiding_questions(content: str) -> List[str]:
    """解析 API 返回的引导性问题"""
    try:
        # 尝试直接解析 JSON
        data = json.loads(content)
        if "guiding_questions" in data:
            return data["guiding_questions"]

        # 尝试从 markdown 代码块中提取
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            json_str = content[start:end].strip()
            data = json.loads(json_str)
            if "guiding_questions" in data:
                return data["guiding_questions"]

        # 尝试从文本中提取 JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            data = json.loads(json_str)
            if "guiding_questions" in data:
                return data["guiding_questions"]

        print(f"  Warning: Could not parse guiding questions from content")
        return []

    except Exception as e:
        print(f"  Error parsing guiding questions: {str(e)}")
        return []


def generate_for_sample(sample: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """为单个样本生成引导性问题"""
    # 提取问题和答案
    problem = sample.get("problem", "")
    # 清理问题文本
    problem = problem.replace("<image>", "").strip()
    if "\n\nOnly focus on the objects inside the red bounding box" in problem:
        problem = problem.split("\n\nOnly focus on the objects inside the red bounding box")[0]

    answer = sample.get("answer", "")

    # 生成 prompt
    prompt = GENERATION_PROMPT.format(question=problem, answer=answer)

    # 调用 API
    result = call_deepseek_api(prompt, api_key)

    if result["success"]:
        questions = parse_guiding_questions(result["content"])
        return {
            "success": True,
            "guiding_questions": questions,
            "raw_response": result["content"]
        }
    else:
        return {
            "success": False,
            "error": result["error"],
            "guiding_questions": []
        }


def main():
    args = parse_args()

    # 读取输入数据
    print(f"Reading input file: {args.input_file}")
    samples = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Total samples: {len(samples)}")

    # 确定处理范围
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else min(start_idx + args.batch_size, len(samples))

    print(f"Processing samples {start_idx} to {end_idx} (total: {end_idx - start_idx})")

    # 创建输出目录
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 处理样本
    results = []
    success_count = 0
    fail_count = 0

    for idx in range(start_idx, end_idx):
        sample = samples[idx]
        print(f"[{idx - start_idx + 1}/{end_idx - start_idx}] Processing sample {idx}...", end=" ")

        result = generate_for_sample(sample, args.api_key)

        if result["success"]:
            success_count += 1
            print(f"✓ ({len(result['guiding_questions'])} questions)")
        else:
            fail_count += 1
            print(f"✗ ({result['error']})")

        results.append({
            "idx": idx,
            "original_sample": sample,
            "guiding_questions": result["guiding_questions"],
            "success": result["success"],
            "error": result.get("error"),
            "raw_response": result.get("raw_response")
        })

        # 避免 API 限流
        time.sleep(0.5)

    # 保存结果
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印统计
    print(f"\n{'=' * 70}")
    print(f"Generation complete!")
    print(f"Success: {success_count}/{end_idx - start_idx} ({100 * success_count / (end_idx - start_idx):.1f}%)")
    print(f"Failed: {fail_count}/{end_idx - start_idx}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
