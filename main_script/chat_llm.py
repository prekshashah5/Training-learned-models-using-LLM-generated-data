import os
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage


def build_chat_llm(
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float = 0.2,
):
    """
    Creates a ChatHuggingFace LLM with controlled generation settings.
    """

    text_gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,                 # ← force greedy
        temperature=temperature,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return ChatHuggingFace(llm=llm)


def run_chat_prompt(chat_llm, prompt: str) -> str:
    """
    Runs a single-turn chat prompt and returns raw text.
    Enforces JSON-only behavior via system message.
    """

    messages = [
        SystemMessage(content="You output valid JSON only. No extra text."),
        HumanMessage(content=prompt),
    ]

    response = chat_llm.invoke(messages)
    print("[info] Raw LLM response:", response.content.strip())
    return response.content.strip()
