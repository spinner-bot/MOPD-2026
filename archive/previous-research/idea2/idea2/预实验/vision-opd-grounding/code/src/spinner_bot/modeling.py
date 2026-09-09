from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def load_model(model_id: str, device: str | None = None):
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if device is not None:
        kwargs.pop("device_map")
        kwargs["device_map"] = {"": device}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, local_files_only=True, **kwargs
    )
    model.eval()
    return model, processor


def build_messages(prompt: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def infer(model, processor, prompt: str, images: list[Image.Image], max_new_tokens: int = 32) -> str:
    messages = build_messages(prompt, images)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()
