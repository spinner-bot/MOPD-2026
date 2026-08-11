"""GRPO-Guard with multi-dataset training.

Combines the GRPO-Guard training recipe (``rationorm`` ratio
correction, ``proxy_advantage_ratio`` advantage mixing, EMA) with
per-dataset round-robin sampling + per-dataset stat tracking. A single
run trains on multiple prompt sources (e.g. PickScore + OCR + GenEval),
with each batch's matching scorer driving that batch's advantage so the
OCR scorer is never run on PickScore prompts (which would crash on
``prompt.split('"')[1]``) and vice versa.
"""

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
import wandb
from functools import partial, wraps
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def make_no_grad(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper


def _build_datasets_config(config):
    """Normalize single- vs multi-dataset configs into a single list.

    Multi-dataset mode: ``config.train.datasets`` is a list of dicts,
    each with keys ``name``, ``dataset``, ``reward_fn`` and optionally
    ``prompt_fn`` and ``proxy_reward``. Each entry pairs a prompt source
    with its own training reward dict (``reward_fn``) and proxy scorer
    (``proxy_reward``, feeding ``config.proxy_advantage_ratio`` mixing).
    The sampling loop round-robins between entries and only the matching
    scorer is invoked on each batch's prompts.

    Single-dataset mode: ``config.train.datasets`` is unset; a
    one-element list is synthesized from the legacy top-level fields
    (``config.dataset`` / ``config.reward_fn`` / ``config.proxy_reward_fn``
    / ``config.prompt_fn``).
    """
    global_prompt_fn = getattr(config, "prompt_fn", "general_ocr")
    global_proxy = dict(getattr(config, "proxy_reward_fn", {}) or {})

    datasets = getattr(config.train, "datasets", None)
    if datasets:
        out = []
        for d in datasets:
            entry = dict(d)
            for required in ("name", "dataset", "reward_fn"):
                if required not in entry:
                    raise ValueError(
                        f"Dataset entry {entry!r} is missing required key "
                        f"'{required}'."
                    )
            entry["reward_fn"] = dict(entry["reward_fn"])
            if not entry["reward_fn"]:
                raise ValueError(
                    f"Dataset entry {entry['name']!r} has an empty "
                    f"`reward_fn` dict; advantage computation needs at "
                    f"least one reward."
                )
            # ``proxy_reward`` is optional; fall back to the reward fn.
            if entry.get("proxy_reward") is None:
                entry["proxy_reward"] = dict(entry["reward_fn"])
            else:
                entry["proxy_reward"] = dict(entry["proxy_reward"])
            entry["prompt_fn"] = entry.get("prompt_fn") or global_prompt_fn
            out.append(entry)
        names = [d["name"] for d in out]
        if len(set(names)) != len(names):
            raise ValueError(f"Dataset entry names must be unique (got {names}).")
        return out

    # Single-dataset back-compat path.
    if not getattr(config, "dataset", None):
        raise ValueError(
            "GRPO-Guard multi-dataset training requires either "
            "`config.train.datasets` (multi-dataset) or `config.dataset` "
            "+ `config.reward_fn` (single-dataset back-compat) to be set."
        )
    legacy_reward = dict(getattr(config, "reward_fn", {}) or {})
    if not legacy_reward:
        raise ValueError(
            "Single-dataset back-compat path requires `config.reward_fn` "
            "to be a non-empty dict."
        )
    return [
        {
            "name": "main",
            "dataset": config.dataset,
            "prompt_fn": global_prompt_fn,
            "reward_fn": legacy_reward,
            "proxy_reward": global_proxy or dict(legacy_reward),
        }
    ]


from accelerate.state import AcceleratorState


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)

    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array,
        return_inverse=True,
        return_counts=True,
    )

    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    prompt_std_devs = np.array([np.std(group) for group in reward_groups])

    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    return zero_std_ratio, prompt_std_devs.mean()


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
        return_sqrt_dt=config.rationorm,
    )
    if config.rationorm:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def _eval_one_dataset(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    reward_fn,
    proxy_reward_fn,
    executor,
    autocast,
    log_prefix,
    save_subdir,
):
    """Run eval over a single dataset's test dataloader + matching scorers.

    EMA copy in/out is the caller's responsibility — this function
    leaves the model state unchanged.
    """
    eval_guidance_scale = getattr(
        config.sample, "eval_guidance_scale", config.sample.guidance_scale
    )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    last_images = None
    last_prompts = None
    last_rewards = {}
    last_proxy_rewards = {}
    for test_batch in tqdm(
        test_dataloader,
        desc=f"Eval[{log_prefix or 'main'}]: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device,
        )
        if len(prompt_embeds) < len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=eval_guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        rewards_fut = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        if proxy_reward_fn is not None:
            proxy_fut = executor.submit(proxy_reward_fn, images, prompts, prompt_metadata, only_strict=False)
        else:
            proxy_fut = None
        time.sleep(0)
        rewards, _ = rewards_fut.result()
        if proxy_fut is not None:
            proxy_rewards, _ = proxy_fut.result()
        else:
            proxy_rewards = {}

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
        for key, value in proxy_rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[f"proxy_{key}"].append(rewards_gather)

        last_images = images
        last_prompts = prompts
        last_rewards = rewards
        last_proxy_rewards = proxy_rewards

    if last_images is None:
        return

    last_batch_images_gather = accelerator.gather(
        torch.as_tensor(last_images, device=accelerator.device)
    ).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        last_prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in last_rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(
            torch.as_tensor(value, device=accelerator.device)
        ).cpu().numpy()
    for key, value in last_proxy_rewards.items():
        last_batch_rewards_gather[f"proxy_{key}"] = accelerator.gather(
            torch.as_tensor(value, device=accelerator.device)
        ).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [
                {k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather}
                for index in sample_indices
            ]
            for key, value in all_rewards.items():
                print(f"{log_prefix}{key}", value.shape)
            wandb.log(
                {
                    f"eval_{log_prefix}images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(
                                f"{k}: {v:.2f}" for k, v in reward.items() if v != -10
                            ),
                        )
                        for idx, (prompt, reward) in enumerate(
                            zip(sampled_prompts, sampled_rewards)
                        )
                    ],
                    **{
                        f"eval_{log_prefix}reward_{key}": np.mean(value[value != -10])
                        for key, value in all_rewards.items()
                    },
                },
                step=global_step,
            )
            save_eval_dir = getattr(config, "save_eval_dir", None)
            if save_eval_dir:
                save_dir = os.path.join(save_eval_dir, f"step_{global_step:04d}")
                if save_subdir:
                    save_dir = os.path.join(save_dir, save_subdir)
                os.makedirs(save_dir, exist_ok=True)
                for idx in range(num_samples):
                    image = last_batch_images_gather[idx]
                    pil = Image.fromarray(
                        (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil.save(os.path.join(save_dir, f"image_{idx:04d}.png"))


def eval(
    pipeline,
    test_dataloaders,
    datasets_cfg,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    reward_fns_per_dataset,
    proxy_reward_fns_per_dataset,
    executor,
    autocast,
    num_train_timesteps,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    use_prefix = len(datasets_cfg) > 1
    for dc, loader, reward_fn, proxy_reward_fn in zip(
        datasets_cfg,
        test_dataloaders,
        reward_fns_per_dataset,
        proxy_reward_fns_per_dataset,
    ):
        log_prefix = f"{dc['name']}_" if use_prefix else ""
        save_subdir = dc["name"] if use_prefix else None
        _eval_one_dataset(
            pipeline=pipeline,
            test_dataloader=loader,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            config=config,
            accelerator=accelerator,
            global_step=global_step,
            reward_fn=reward_fn,
            proxy_reward_fn=proxy_reward_fn,
            executor=executor,
            autocast=autocast,
            log_prefix=log_prefix,
            save_subdir=save_subdir,
        )

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="flow_grpo",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)

    pipeline.transformer.to(accelerator.device)

    datasets_cfg = _build_datasets_config(config)
    if accelerator.is_local_main_process:
        logger.info(
            f"GRPO-Guard MultiDataset: configured {len(datasets_cfg)} dataset entry/entries:"
        )
        for dc in datasets_cfg:
            logger.info(
                f"  entry {dc['name']}: prompts={dc['dataset']}, "
                f"prompt_fn={dc['prompt_fn']}, "
                f"reward_fn={dc['reward_fn']}, "
                f"proxy_reward={dc['proxy_reward']}"
            )

    # Per-dataset reward fn (drives the advantage) + per-dataset proxy
    # reward fn (monitoring + ``proxy_advantage_ratio`` mixing). Built
    # before LoRA so the AcceleratorState hack stays isolated, and one
    # scorer per dataset so each batch invokes only its matching pair.
    ## IMPORTANT: make sure reward fn is created before we create lora model
    original_reset = AcceleratorState._reset_state
    AcceleratorState._reset_state = lambda *args, **kwargs: None
    reward_fns_per_dataset = []
    proxy_reward_fns_per_dataset = []
    for dc in datasets_cfg:
        rfn = getattr(flow_grpo.rewards, "multi_score")(
            accelerator.device, dc["reward_fn"]
        )
        rfn = make_no_grad(rfn)
        reward_fns_per_dataset.append(rfn)
        if dc["proxy_reward"]:
            pfn = getattr(flow_grpo.rewards, "multi_score")(
                accelerator.device, dc["proxy_reward"]
            )
            pfn = make_no_grad(pfn)
        else:
            pfn = None
        proxy_reward_fns_per_dataset.append(pfn)
    AcceleratorState._reset_state = original_reset
    torch.cuda.empty_cache()

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Each entry's ``prompt_fn`` selects which Dataset class loads its
    # prompts, so a single run can mix prompt sources with different
    # on-disk schemas (``general_ocr`` -> TextPromptDataset,
    # ``geneval`` -> GenevalPromptDataset).
    def _resolve_dataset_cls(prompt_fn):
        if prompt_fn == "general_ocr":
            return TextPromptDataset
        elif prompt_fn == "geneval":
            return GenevalPromptDataset
        else:
            raise NotImplementedError(
                "Only `general_ocr` and `geneval` prompt_fns are supported."
            )

    dataset_classes = [
        _resolve_dataset_cls(dc["prompt_fn"]) for dc in datasets_cfg
    ]

    train_datasets = [
        cls(dc["dataset"], "train")
        for cls, dc in zip(dataset_classes, datasets_cfg)
    ]
    test_datasets = [
        cls(dc["dataset"], "test")
        for cls, dc in zip(dataset_classes, datasets_cfg)
    ]

    train_samplers = []
    train_dataloaders = []
    for cls, ds in zip(dataset_classes, train_datasets):
        sampler = DistributedKRepeatSampler(
            dataset=ds,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=1,
            collate_fn=cls.collate_fn,
        )
        train_samplers.append(sampler)
        train_dataloaders.append(loader)

    test_dataloaders = [
        DataLoader(
            ds,
            batch_size=config.sample.test_batch_size,
            collate_fn=cls.collate_fn,
            shuffle=False,
            num_workers=8,
        )
        for cls, ds in zip(dataset_classes, test_datasets)
    ]

    if len(datasets_cfg) > 1:
        assert config.sample.num_batches_per_epoch % len(datasets_cfg) == 0, (
            f"`config.sample.num_batches_per_epoch` "
            f"(={config.sample.num_batches_per_epoch}) must be divisible by "
            f"the number of dataset entries (={len(datasets_cfg)}) so each "
            f"dataset gets the same number of sampling batches per epoch."
        )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # Per-dataset stat trackers — each dataset's reward distribution
    # lives on a different scale, so we whiten each independently rather
    # than mixing scales through a single tracker. Each dataset also has
    # its own proxy tracker for the ``proxy_advantage_ratio`` mixing.
    if config.per_prompt_stat_tracking:
        stat_trackers = [
            PerPromptStatTracker(config.sample.global_std)
            for _ in datasets_cfg
        ]
        proxy_stat_trackers = [
            PerPromptStatTracker(config.sample.global_std)
            for _ in datasets_cfg
        ]
    else:
        stat_trackers = None
        proxy_stat_trackers = None

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare with `accelerator`. Unpack manually so `train_dataloaders`
    # and `test_dataloaders` (one per dataset) stay as lists.
    prepared = accelerator.prepare(
        transformer,
        optimizer,
        *train_dataloaders,
        *test_dataloaders,
    )
    n_d = len(datasets_cfg)
    transformer = prepared[0]
    optimizer = prepared[1]
    train_dataloaders = list(prepared[2 : 2 + n_d])
    test_dataloaders = list(prepared[2 + n_d : 2 + 2 * n_d])

    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    # One infinite iterator per dataset; the sampling loop picks the
    # dataset (prompt source + reward fns) via ``i % len(datasets_cfg)``.
    train_iters = [iter(loader) for loader in train_dataloaders]

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0 and epoch > 0:
            eval(
                pipeline,
                test_dataloaders,
                datasets_cfg,
                text_encoders,
                tokenizers,
                config,
                accelerator,
                global_step,
                reward_fns_per_dataset,
                proxy_reward_fns_per_dataset,
                executor,
                autocast,
                num_train_timesteps,
                ema,
                transformer_trainable_parameters,
            )
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # Round-robin: batch ``i`` uses dataset entry
            # ``i % len(datasets_cfg)``, selecting the prompt source and
            # the matching reward / proxy reward functions.
            dataset_idx = i % len(datasets_cfg)
            train_samplers[dataset_idx].set_epoch(
                epoch * config.sample.num_batches_per_epoch + i
            )
            prompts, prompt_metadata = next(train_iters[dataset_idx])

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device,
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch * 10000 + i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    if config.rationorm:
                        images, latents, log_probs, prev_sample_mean = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            generator=generator,
                            return_prev_sample_mean=True,
                        )
                    else:
                        images, latents, log_probs = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            generator=generator,
                        )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
            if config.rationorm:
                prev_sample_mean = torch.stack(prev_sample_mean, dim=1)
            else:
                prev_sample_mean = log_probs

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )

            # Per-batch reward & proxy futures using only the matching
            # dataset's scorers, so the OCR scorer never sees PickScore
            # prompts and vice versa.
            batch_reward_fn = reward_fns_per_dataset[dataset_idx]
            batch_proxy_fn = proxy_reward_fns_per_dataset[dataset_idx]
            rewards = executor.submit(
                batch_reward_fn, images, prompts, prompt_metadata, only_strict=True
            )
            if batch_proxy_fn is not None:
                proxy_rewards = executor.submit(
                    batch_proxy_fn,
                    images,
                    prompts,
                    prompt_metadata,
                    only_strict=False,
                )
            else:
                proxy_rewards = None
            time.sleep(0)

            # Per-sample integer tag identifying the dataset entry that
            # produced this batch, carried through rebatching so the
            # per-dataset stat tracker can pick out the right rows.
            B = prompt_embeds.shape[0]
            dataset_idx_tensor = torch.full(
                (B,), dataset_idx, dtype=torch.long, device=accelerator.device
            )

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "prev_sample_mean": prev_sample_mean,
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "proxy_rewards": proxy_rewards,
                    "dataset_idx": dataset_idx_tensor,
                }
            )

        # Resolve reward / proxy futures. Each batch's reward dict has
        # only that dataset's keys; pad each sample's reward dict to the
        # union of all keys with a `-10` sentinel so the concat logic
        # can produce a single epoch-wide reward dict. Downstream wandb
        # logging filters `-10` before averaging.
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, _ = sample["rewards"].result()
            if sample["proxy_rewards"] is not None:
                proxy_rewards, _ = sample["proxy_rewards"].result()
            else:
                proxy_rewards = {}
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            sample["proxy_rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in proxy_rewards.items()
            }

        all_reward_keys = sorted({k for s in samples for k in s["rewards"].keys()})
        all_proxy_keys = sorted({k for s in samples for k in s["proxy_rewards"].keys()})
        for s in samples:
            n = s["timesteps"].shape[0]
            for k in all_reward_keys:
                if k not in s["rewards"]:
                    s["rewards"][k] = torch.full(
                        (n,), -10.0, device=accelerator.device, dtype=torch.float32
                    )
            for k in all_proxy_keys:
                if k not in s["proxy_rewards"]:
                    s["proxy_rewards"][k] = torch.full(
                        (n,), -10.0, device=accelerator.device, dtype=torch.float32
                    )

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, ii in enumerate(sample_indices):
                    image = images[ii]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[ii] for ii in sample_indices]
                # The collated reward tensors span the whole epoch; the
                # preview uses only the last sampled batch, so slice off
                # the last B entries of the collated "avg" tensors.
                last_avg = samples["rewards"]["avg"][-len(images):]
                if "avg" in samples["proxy_rewards"]:
                    last_proxy = samples["proxy_rewards"]["avg"][-len(images):]
                else:
                    last_proxy = None
                sampled_rewards = [last_avg[ii].item() for ii in sample_indices]
                if last_proxy is not None:
                    sampled_proxy_rewards = [last_proxy[ii].item() for ii in sample_indices]
                else:
                    sampled_proxy_rewards = [float("nan")] * len(sample_indices)

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f} | proxy_avg: {proxy_avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward, proxy_avg_reward) in enumerate(
                                zip(sampled_prompts, sampled_rewards, sampled_proxy_rewards)
                            )
                        ],
                    },
                    step=global_step,
                )

        # GRPO-Guard reward bookkeeping:
        # - `ori_avg`: 1D per-sample reward for the zero-std diagnostic
        # - `avg`:     2D (B, T) reward used by `stat_tracker.update`
        # - `proxy_avg`: 2D (B, T) proxy reward for the
        #   ``proxy_advantage_ratio`` advantage mixing.
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        if "avg" in samples["proxy_rewards"]:
            samples["rewards"]["proxy_avg"] = (
                samples["proxy_rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
            )
        else:
            # No proxy scorer configured — feed a zero stand-in so the
            # proxy-advantage path stays well-defined; with
            # `proxy_advantage_ratio = 0` it contributes nothing anyway.
            samples["rewards"]["proxy_avg"] = torch.zeros_like(samples["rewards"]["avg"])
        del samples["proxy_rewards"]

        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # Also gather the per-sample dataset index so the per-dataset
        # stat tracker can pick out the matching rows below.
        gathered_dataset_idx = (
            accelerator.gather(samples["dataset_idx"]).cpu().numpy()
        )

        # log rewards, filtering the `-10` sentinel used to pad
        # off-source keys so each `reward_<key>` stays an honest mean.
        if accelerator.is_main_process:
            reward_log = {"epoch": epoch}
            for key, value in gathered_rewards.items():
                if "_strict_accuracy" in key or "_accuracy" in key:
                    continue
                valid = value[value != -10] if value.ndim == 1 else value[value[:, 0] != -10]
                if valid.size > 0:
                    reward_log[f"reward_{key}"] = float(valid.mean())
            wandb.log(reward_log, step=global_step)

        # per-prompt mean/std tracking — done PER-DATASET so reward
        # distributions on different scales are whitened independently.
        if config.per_prompt_stat_tracking:
            prompt_ids_np = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts_str = pipeline.tokenizer.batch_decode(
                prompt_ids_np, skip_special_tokens=True
            )
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts_str))
                print("len unique prompts", len(set(prompts_str)))

            advantages = np.zeros_like(gathered_rewards['avg'])
            for d_idx, _ in enumerate(datasets_cfg):
                mask = gathered_dataset_idx == d_idx
                if not mask.any():
                    continue
                idxs = np.where(mask)[0]
                sub_prompts = [prompts_str[i] for i in idxs]
                sub_rewards = gathered_rewards['avg'][idxs]
                sub_proxy = gathered_rewards['proxy_avg'][idxs]

                main_adv = stat_trackers[d_idx].update(sub_prompts, sub_rewards)
                stat_trackers[d_idx].clear()

                if config.proxy_advantage_ratio != 0.0:
                    proxy_adv = proxy_stat_trackers[d_idx].update(
                        sub_prompts, sub_proxy
                    )
                    proxy_stat_trackers[d_idx].clear()
                    advantages[idxs] = (
                        main_adv + config.proxy_advantage_ratio * proxy_adv
                    )
                else:
                    advantages[idxs] = main_adv

            group_size_total = 0.0
            trained_prompt_total = 0
            for tracker in stat_trackers:
                gs, tp = tracker.get_stats()
                group_size_total += gs
                trained_prompt_total += tp
            group_size = group_size_total / max(1, len(stat_trackers))

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                prompts_str, gathered_rewards
            )
            proxy_avg_1d = gathered_rewards['proxy_avg'][:, 0]
            valid_proxy = proxy_avg_1d[proxy_avg_1d != -10]
            proxy_reward_std_mean = (
                float(valid_proxy.std()) if valid_proxy.size > 0 else 0.0
            )

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_total,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "proxy_reward_std_mean": proxy_reward_std_mean,
                    },
                    step=global_step,
                )
        else:
            # No per-prompt stat tracking — whiten per-dataset using the
            # gathered means/stds. Still respect proxy_advantage_ratio.
            advantages = np.zeros_like(gathered_rewards['avg'])
            for d_idx, _ in enumerate(datasets_cfg):
                mask = gathered_dataset_idx == d_idx
                if not mask.any():
                    continue
                sub = gathered_rewards['avg'][mask]
                main_adv = (sub - sub.mean()) / (sub.std() + 1e-4)
                if config.proxy_advantage_ratio != 0.0:
                    sub_proxy = gathered_rewards['proxy_avg'][mask]
                    proxy_adv = (sub_proxy - sub_proxy.mean()) / (sub_proxy.std() + 1e-4)
                    advantages[mask] = (
                        main_adv + config.proxy_advantage_ratio * proxy_adv
                    )
                else:
                    advantages[mask] = main_adv

        # ungather advantages; we only need the entries corresponding to this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)

        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0 or true_count == 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item() // config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = compute_log_prob(
                                transformer, pipeline, sample, j, embeds, pooled_embeds, config
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _, _ = compute_log_prob(
                                            transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                        )

                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        if config.rationorm:
                            sigma_t = std_dev_t.mean()
                            ratio_mean_bias = (prev_sample_mean - sample["prev_sample_mean"][:, j]).pow(2).mean(
                                dim=tuple(range(1, log_prob.ndim))
                            )
                            ratio_mean_bias = ratio_mean_bias / (2 * (sqrt_dt.mean() * sigma_t) ** 2)
                            ratio = torch.exp(
                                (log_prob - sample["log_probs"][:, j] + ratio_mean_bias)
                                * (sqrt_dt.mean() * sigma_t)
                            )
                        else:
                            ratio = torch.exp(log_prob - sample["log_probs"][:, j])

                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        if config.rationorm:
                            policy_loss = policy_loss / (sqrt_dt.mean() ** 2)

                        if config.train.beta > 0:
                            kl_loss = (
                                (prev_sample_mean - prev_sample_mean_ref) ** 2
                            ).mean(dim=(1, 2, 3), keepdim=True) / (
                                2 * std_dev_t ** 2
                            )
                            kl_loss = torch.mean(kl_loss.reshape(-1))
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch += 1


if __name__ == "__main__":
    app.run(main)
