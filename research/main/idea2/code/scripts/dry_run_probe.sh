#!/bin/bash

# Dry run script for probe logging verification
# Uses smaller models and minimal data for quick testing

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Use smaller models available on server
MODEL_PATH="/root/hjc/models/Qwen2.5-0.5B-Instruct"
TEXT_TEACHER_PATH="/root/hjc/models/Qwen2.5-0.5B-Instruct"

# Create minimal test data
DATA_DIR="${PROJECT_ROOT}/data"
mkdir -p "$DATA_DIR"

echo "Creating minimal test data..."
python3 << EOF
import pandas as pd
import os

# Create minimal parquet file with 2 samples
data = {
    'prompt': ['What is 2+2?', 'What is the capital of France?'],
    'images': [[], []],
    'ground_truth': ['4', 'Paris']
}

df = pd.DataFrame(data)
os.makedirs('$DATA_DIR', exist_ok=True)
df.to_parquet('$DATA_DIR/train.parquet', index=False)
print(f"Created test data at $DATA_DIR/train.parquet")
EOF

echo "Starting dry run with probe logging..."
echo "Model: $MODEL_PATH"
echo "Text teacher: $TEXT_TEACHER_PATH"

python3 -m verl.trainer.main_ppo --config-name vopd \
    data.train_files="[\"${DATA_DIR}/train.parquet\"]" \
    data.val_files="[]" \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=128 \
    data.max_response_length=32 \
    data.truncation=error \
    data.shuffle=False \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.train_batch_size=2 \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=256 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=10 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=128 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=legacy \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images \
    +actor_rollout_ref.actor.self_distillation.text_teacher_path=$TEXT_TEACHER_PATH \
    +actor_rollout_ref.actor.self_distillation.mcd_alpha=0.5 \
    +actor_rollout_ref.actor.self_distillation.mcd_beta=0.5 \
    +actor_rollout_ref.actor.self_distillation.mcd_tau=1.0 \
    +actor_rollout_ref.actor.self_distillation.mcd_epsilon=0.1 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.alpha=0.5 \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=256 \
    actor_rollout_ref.rollout.max_model_len=256 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
    actor_rollout_ref.rollout.response_length=32 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.enable=False \
    critic.model.path=$MODEL_PATH \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    trainer.project_name=idea2_dry_run \
    trainer.group_name=probe_test \
    trainer.experiment_name=probe_test \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${PROJECT_ROOT}/checkpoints/dry_run" \
    trainer.rollout_data_dir="${PROJECT_ROOT}/rollouts/dry_run" \
    actor_rollout_ref.actor.probe_logging.enabled=true \
    actor_rollout_ref.actor.probe_logging.interval=1 \
    actor_rollout_ref.actor.probe_logging.probe_batch_path="${PROJECT_ROOT}/probe_data/probe_batch.jsonl" \
    actor_rollout_ref.actor.probe_logging.output_dir="${PROJECT_ROOT}/probe_logs"

echo "Dry run completed. Checking probe logs..."
ls -lh ${PROJECT_ROOT}/probe_logs/ || echo "No probe logs generated"
