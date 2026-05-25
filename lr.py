"""Utility for creating and saving multiple LoRA adapters from a base model."""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration


def build_target_modules(layer_start: int, layer_end: int) -> list[str]:
    """Return a list of module paths for the given layer range (inclusive)."""
    modules = []
    for i in range(layer_start, layer_end + 1):
        modules.extend([
            f"model.language_model.layers.{i}.self_attn.q_proj",
            f"model.language_model.layers.{i}.self_attn.k_proj",
            f"model.language_model.layers.{i}.self_attn.v_proj",
            f"model.language_model.layers.{i}.self_attn.o_proj",
            f"model.language_model.layers.{i}.mlp.gate_proj",
            f"model.language_model.layers.{i}.mlp.up_proj",
            f"model.language_model.layers.{i}.mlp.down_proj",
        ])
    return modules


def load_model(model_id: str) -> Qwen3VLForConditionalGeneration:
    """Load the base model from a local path or HuggingFace hub."""
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def create_adapters(
    model,
    configs: list[dict],
    target_modules: list[str],
    output_base: str,
) -> None:
    """Create and save one LoRA adapter per config entry, resetting the model between runs."""
    for i, cfg in enumerate(tqdm(configs, desc="Creating adapters")):
        peft_config = LoraConfig(
            r=cfg["r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.0),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

        model = get_peft_model(model, peft_config)

        adapter_dir = f"{output_base}/lora_adapter_{i}"
        model.save_pretrained(adapter_dir)
        print(f"Adapter {i} saved to {adapter_dir}")

        model = model.merge_and_unload()
