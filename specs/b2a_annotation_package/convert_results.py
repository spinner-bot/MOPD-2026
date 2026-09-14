#!/usr/bin/env python3
"""
B2a 标注结果转换脚本

功能：
将 HTML 标注工具导出的 JSON 结果转换为项目标准格式（与 blind-review-32.jsonl 对应）

使用方法：
    python convert_results.py <输入JSON文件> [输出JSONL文件]

示例：
    python convert_results.py b2a_annotation_results_2026-09-14.json
    python convert_results.py results.json output.jsonl
"""

import json
import sys
import os
from datetime import datetime

def convert_annotation_to_standard_format(annotation, original_record):
    """
    将 HTML 导出的标注转换为项目标准格式

    Args:
        annotation: HTML 导出的单条标注（dict）
        original_record: blind-review-32.jsonl 中的原始记录（dict）

    Returns:
        标准格式的记录（dict）
    """
    # 复制原始记录的元数据字段
    result = {
        # 保留原始标识
        "calibration_record_id": original_record.get("calibration_record_id", ""),
        "calibration_source_index": original_record.get("calibration_source_index", 0),
        "image_candidate_id": original_record.get("image_candidate_id", ""),
        "image_path": original_record.get("image_path", ""),
        "image_sha256": original_record.get("image_sha256", ""),
        "source_dataset": original_record.get("source_dataset", ""),
        "source_image_id": original_record.get("source_image_id", ""),

        # 保留盲审控制字段
        "machine_proposal_exposed": original_record.get("machine_proposal_exposed", False),
        "review_phase": original_record.get("review_phase", "phase_1_blind_image_only"),

        # 人工标注字段（从 HTML 导出）
        "human_image_usable": annotation.get("human_image_usable", False),
        "human_affordance_decision": annotation.get("human_affordance_decision", ""),
        "human_affordance_family": annotation.get("human_affordance_family", ""),
        "human_target_question": annotation.get("human_target_question", ""),
        "human_canonical_answer": annotation.get("human_canonical_answer", ""),
        "human_evidence_descriptions": annotation.get("human_evidence_descriptions", []),
        "human_reasoning_steps": annotation.get("human_reasoning_steps", []),
        "human_difficult_visual_evidence": annotation.get("human_difficult_visual_evidence", []),
        "human_external_knowledge_needed": annotation.get("human_external_knowledge_needed", False),
        "human_notes": annotation.get("human_notes", ""),
        "human_reviewer": annotation.get("human_reviewer", ""),
        "human_reviewed_at": annotation.get("human_reviewed_at", ""),
        "human_verified": annotation.get("human_verified", False),

        # 更新审核状态
        "review_status": "human_blind_affordance_review_complete"
    }

    return result

def load_original_records(jsonl_path):
    """
    加载原始的 blind-review-32.jsonl 文件

    Returns:
        dict: image_name -> original_record
    """
    records = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            # 从 image_path 提取文件名
            image_name = os.path.basename(record['image_path'])
            records[image_name] = record
    return records

def convert_file(input_json_path, output_jsonl_path=None):
    """
    转换整个标注文件

    Args:
        input_json_path: HTML 导出的 JSON 文件路径
        output_jsonl_path: 输出的 JSONL 文件路径（可选，默认生成）
    """
    # 生成输出文件名
    if output_jsonl_path is None:
        base_name = os.path.splitext(input_json_path)[0]
        output_jsonl_path = f"{base_name}_standard.jsonl"

    # 查找原始的 blind-review-32.jsonl
    # 假设它在项目的标准位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
    original_jsonl_path = os.path.join(
        project_root,
        'research', 'main', 'outputs', 'runs',
        '20260914T0328-image-affordance-b2a-blind-calibration',
        'outputs', 'blind-review-32.jsonl'
    )

    if not os.path.exists(original_jsonl_path):
        print(f"警告：找不到原始文件 {original_jsonl_path}")
        print("将尝试在当前目录查找 blind-review-32.jsonl")

        # 尝试在当前目录查找
        alt_path = os.path.join(script_dir, 'blind-review-32.jsonl')
        if os.path.exists(alt_path):
            original_jsonl_path = alt_path
        else:
            print("错误：找不到 blind-review-32.jsonl")
            print("请将此脚本和 blind-review-32.jsonl 放在同一目录，或指定完整路径")
            return False

    print(f"加载原始记录：{original_jsonl_path}")
    original_records = load_original_records(original_jsonl_path)
    print(f"  找到 {len(original_records)} 条原始记录")

    # 加载 HTML 导出的标注
    print(f"加载标注结果：{input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    print(f"  找到 {len(annotations)} 条标注")

    # 转换
    print("开始转换...")
    converted = []
    missing = []

    for annotation in annotations:
        image_name = annotation.get('image_name', '')

        if image_name not in original_records:
            missing.append(image_name)
            print(f"  ⚠ 找不到原始记录：{image_name}")
            continue

        original = original_records[image_name]
        result = convert_annotation_to_standard_format(annotation, original)
        converted.append(result)

    print(f"  成功转换：{len(converted)} 条")
    if missing:
        print(f"  缺失原始记录：{len(missing)} 条")

    # 写入输出文件
    print(f"写入输出文件：{output_jsonl_path}")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✓ 转换完成！")
    print(f"  输出文件：{output_jsonl_path}")
    print(f"  记录数：{len(converted)}")

    return True

def main():
    if len(sys.argv) < 2:
        print("用法：python convert_results.py <输入JSON文件> [输出JSONL文件]")
        print("\n示例：")
        print("  python convert_results.py b2a_annotation_results.json")
        print("  python convert_results.py results.json output.jsonl")
        sys.exit(1)

    input_json = sys.argv[1]
    output_jsonl = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_json):
        print(f"错误：找不到输入文件 {input_json}")
        sys.exit(1)

    success = convert_file(input_json, output_jsonl)

    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()
