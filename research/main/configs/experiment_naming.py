"""
实验命名规范

按实验方案 §79 定义统一命名：

BASE-Qwen35-4B

OPD00-Qwen35-4B
OPDV-Qwen35-4B
OPDTQ-Qwen35-4B
OPDTQA-Qwen35-4B
OPDVT-Qwen35-4B

ABL-T-RANDOM
ABL-T-SHUFFLE
ABL-T-GENERIC
ABL-T-PARTIAL

ABL-V-SHAM

FUSION-VT-SIM
FUSION-VT-ALT
FUSION-VT-DUAL050
"""

from dataclasses import dataclass
from typing import Literal


# === 基础模型 ===
BASE_MODELS = {
    "4B": "Qwen35-4B",
    "9B": "Qwen35-9B",
}

# === 主实验命名 ===
MAIN_EXPERIMENTS = {
    "base": "BASE-{model}",
    "opd00": "OPD00-{model}",  # Vanilla OPD, no privilege
    "opdv": "OPDV-{model}",    # Visual privilege only
    "opdtq": "OPDTQ-{model}",  # Text-Q privilege
    "opdtqa": "OPDTQA-{model}",  # Text-QA privilege
    "opdvt": "OPDVT-{model}",  # Visual + Text privilege
}

# === 消融实验命名 ===
ABLATION_EXPERIMENTS = {
    # Text ablations
    "abl_t_random": "ABL-T-RANDOM-{model}",
    "abl_t_shuffle": "ABL-T-SHUFFLE-{model}",
    "abl_t_generic": "ABL-T-GENERIC-{model}",
    "abl_t_partial": "ABL-T-PARTIAL-{model}",

    # Visual ablations
    "abl_v_sham": "ABL-V-SHAM-{model}",
}

# === 融合实验命名 ===
FUSION_EXPERIMENTS = {
    "fusion_vt_sim": "FUSION-VT-SIM-{model}",      # Simultaneous teacher
    "fusion_vt_alt": "FUSION-VT-ALT-{model}",      # Alternating teacher
    "fusion_vt_dual025": "FUSION-VT-DUAL025-{model}",  # Dual teacher, λ=0.25
    "fusion_vt_dual050": "FUSION-VT-DUAL050-{model}",  # Dual teacher, λ=0.50
    "fusion_vt_dual075": "FUSION-VT-DUAL075-{model}",  # Dual teacher, λ=0.75
}


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    experiment_type: str  # "main", "ablation", "fusion"
    model_size: Literal["4B", "9B"]
    seed: int
    dataset_version: str
    git_commit: str


def get_experiment_name(
    experiment_id: str,
    model_size: str = "4B"
) -> str:
    """
    获取实验名称

    Args:
        experiment_id: 实验 ID（如 "opd00", "opdtq", "abl_t_random"）
        model_size: 模型大小（"4B" 或 "9B"）

    Returns:
        完整实验名称
    """
    model = BASE_MODELS.get(model_size, model_size)

    # 查找命名模板
    if experiment_id in MAIN_EXPERIMENTS:
        template = MAIN_EXPERIMENTS[experiment_id]
    elif experiment_id in ABLATION_EXPERIMENTS:
        template = ABLATION_EXPERIMENTS[experiment_id]
    elif experiment_id in FUSION_EXPERIMENTS:
        template = FUSION_EXPERIMENTS[experiment_id]
    else:
        return f"{experiment_id.upper()}-{model}"

    return template.format(model=model)


def get_stage0_condition_names(model_size: str = "4B") -> dict[str, str]:
    """
    获取 Stage 0 的所有条件名称（方案 §20）

    Returns:
        {condition_id: full_name}
    """
    model = BASE_MODELS.get(model_size, model_size)

    return {
        "S": f"S-{model}",
        "V_crop": f"V-CROP-{model}",
        "V_add": f"V-ADD-{model}",
        "T_Q": f"T-Q-{model}",
        "T_QA": f"T-QA-{model}",
        "VT": f"VT-{model}",
        "T_random": f"T-RANDOM-{model}",
        "T_shuffle": f"T-SHUFFLE-{model}",
        "T_generic": f"T-GENERIC-{model}",
    }


def generate_run_id(
    experiment_id: str,
    seed: int,
    timestamp: str = None
) -> str:
    """
    生成 run ID

    Args:
        experiment_id: 实验 ID
        seed: 随机种子
        timestamp: 时间戳（可选）

    Returns:
        run ID
    """
    if timestamp:
        return f"{experiment_id}_seed{seed}_{timestamp}"
    else:
        return f"{experiment_id}_seed{seed}"


def parse_experiment_name(name: str) -> dict:
    """
    解析实验名称

    Args:
        name: 实验名称（如 "OPDTQ-Qwen35-4B"）

    Returns:
        {
            "experiment_type": "main" / "ablation" / "fusion",
            "experiment_id": "opdtq",
            "model_size": "4B",
            "seed": None
        }
    """
    parts = name.split("-")

    if len(parts) < 2:
        return {"experiment_id": name, "model_size": "unknown"}

    exp_code = parts[0]
    model = parts[1] if len(parts) > 1 else "unknown"

    # 解析模型大小
    model_size = "unknown"
    if "4B" in model:
        model_size = "4B"
    elif "9B" in model:
        model_size = "9B"

    # 解析实验类型
    if exp_code.startswith("ABL"):
        exp_type = "ablation"
    elif exp_code.startswith("FUSION"):
        exp_type = "fusion"
    elif exp_code.startswith("OPD") or exp_code == "BASE":
        exp_type = "main"
    else:
        exp_type = "unknown"

    return {
        "experiment_type": exp_type,
        "experiment_id": exp_code.lower(),
        "model_size": model_size,
    }


# === 实验元数据记录（方案 §79）===

def create_run_metadata(
    experiment_id: str,
    seed: int,
    dataset_version: str,
    git_commit: str,
    config: dict = None
) -> dict:
    """
    创建 run 元数据

    每次 run 记录（方案 §79）：
    - git_commit
    - dataset_version
    - dataset_hash
    - model_checkpoint
    - seed
    - config
    - CUDA/environment
    - training_start_checkpoint
    - evaluation_config
    """
    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "dataset_version": dataset_version,
        "git_commit": git_commit,
        "config": config or {},
        "timestamp": None,  # 运行时填充
        "environment": {
            "cuda_version": None,
            "pytorch_version": None,
            "transformers_version": None,
        }
    }
