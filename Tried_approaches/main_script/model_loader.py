# model_loader.py

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name, cache_dir, device, use_bnb_8bit=False):
    """
    Load model + tokenizer using Hugging Face caching.
    First run downloads, later runs load locally.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    print(f"[info] HF cache dir: {cache_dir.resolve()}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    if use_bnb_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            load_in_8bit=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )

    try:
        model.to(device)
        print(f"[info] Model on device: {device}")
    except Exception as e:
        print(f"[warn] Could not move model to {device}: {e}")

    return tokenizer, model