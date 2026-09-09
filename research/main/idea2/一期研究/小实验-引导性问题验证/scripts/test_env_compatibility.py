#!/usr/bin/env python3
"""
环境兼容性测试脚本

用于快速验证当前环境是否满足 Text-OPD 实验要求。
在服务器上运行此脚本，识别潜在问题。

用法：
    python test_env_compatibility.py
"""

import sys
import subprocess
from typing import Tuple, List


def test_import(module_name: str, expected_version: str = None) -> Tuple[bool, str]:
    """测试模块导入和版本"""
    try:
        module = __import__(module_name, fromlist=[''])
        version = getattr(module, '__version__', 'unknown')

        if expected_version and version != expected_version:
            return False, f"版本不匹配: {version} (期望 {expected_version})"

        return True, f"✓ {module_name} {version}"
    except ImportError as e:
        return False, f"✗ 导入失败: {str(e)}"
    except Exception as e:
        return False, f"✗ 错误: {str(e)}"


def test_cuda_compatibility() -> Tuple[bool, str]:
    """测试 CUDA 兼容性"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "✗ CUDA 不可用"

        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)

        info = f"✓ CUDA {cuda_version}, {gpu_count} GPU(s), 首卡: {gpu_name}"
        return True, info
    except Exception as e:
        return False, f"✗ CUDA 测试失败: {str(e)}"


def test_flash_attn() -> Tuple[bool, str]:
    """测试 flash-attn"""
    try:
        import flash_attn
        version = flash_attn.__version__
        return True, f"✓ flash-attn {version}"
    except ImportError:
        return False, "✗ flash-attn 未安装"
    except Exception as e:
        return False, f"✗ flash-attn 加载失败: {str(e)}"


def test_model_loading() -> Tuple[bool, str]:
    """测试模型加载"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os

        model_path = os.path.expanduser("~/models/Qwen3.5-4B")
        if not os.path.exists(model_path):
            return False, f"✗ 模型路径不存在: {model_path}"

        # 尝试加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        return True, f"✓ 模型加载成功 (tokenizer)"
    except Exception as e:
        return False, f"✗ 模型加载失败: {str(e)}"


def test_data_loading() -> Tuple[bool, str]:
    """测试数据加载"""
    try:
        import datasets
        import os

        data_path = os.path.expanduser("~/idea2_code/code/data/train.parquet")
        if not os.path.exists(data_path):
            return False, f"✗ 数据文件不存在: {data_path}"

        # 检查文件大小
        file_size = os.path.getsize(data_path)
        if file_size < 1000:  # 小于 1KB
            return False, f"✗ 数据文件过小: {file_size} bytes (可能不完整)"

        ds = datasets.load_dataset('parquet', data_files=data_path)
        num_samples = len(ds['train'])

        return True, f"✓ 数据加载成功: {num_samples} 条样本"
    except Exception as e:
        return False, f"✗ 数据加载失败: {str(e)}"


def test_verl_import() -> Tuple[bool, str]:
    """测试 verl 导入"""
    try:
        import verl
        version = verl.__version__
        return True, f"✓ verl {version}"
    except ImportError:
        return False, "✗ verl 未安装"
    except Exception as e:
        return False, f"✗ verl 导入失败: {str(e)}"


def test_vllm_import() -> Tuple[bool, str]:
    """测试 vLLM 导入"""
    try:
        import vllm
        version = vllm.__version__
        return True, f"✓ vllm {version}"
    except ImportError:
        return False, "✗ vllm 未安装"
    except Exception as e:
        return False, f"✗ vllm 导入失败: {str(e)}"


def main():
    """运行所有测试"""
    print("=" * 70)
    print("Text-OPD 环境兼容性测试")
    print("=" * 70)
    print()

    tests = [
        ("Python 版本", lambda: (sys.version_info >= (3, 12), f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")),
        ("PyTorch", lambda: test_import('torch', None)),
        ("CUDA 兼容性", test_cuda_compatibility),
        ("Transformers", lambda: test_import('transformers', None)),
        ("vLLM", test_vllm_import),
        ("verl", test_verl_import),
        ("Flash Attention", test_flash_attn),
        ("模型加载", test_model_loading),
        ("数据加载", test_data_loading),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"测试: {test_name}")
        success, message = test_func()
        results.append((test_name, success, message))
        print(f"  {message}")
        print()

    # 汇总
    print("=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"总计: {passed}/{total} 通过")

    if passed == total:
        print("\n✓ 所有测试通过！环境准备就绪。")
        return 0
    else:
        print("\n✗ 部分测试失败。请根据上述错误信息进行修复。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
