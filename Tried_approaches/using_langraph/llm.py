import torch
from transformers import pipeline

def build_llm(model, tokenizer, device, max_new_tokens, temperature=0.2):
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
    )