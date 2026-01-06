import os
import json
from pathlib import Path
from dotenv import load_dotenv
from transformers import logging as hf_logging

from device import get_best_device
from model_loader import load_model_and_tokenizer

from chat_llm import build_chat_llm, run_chat_prompt
from prompt import render_prompt
from json_utils import coerce_json_array
from state import init_state, render_avoid_block, update_state

# ------------------------ SETUP ------------------------

load_dotenv()
hf_logging.set_verbosity_info()
hf_logging.enable_explicit_format()

device = get_best_device()
print(f"[info] Using device: {device}")

MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
MODEL_FOLDER = Path(os.getenv("MODEL_FOLDER", "../models"))
SCHEMA_FILE = os.getenv("SCHEMA_FILE", "schema.txt")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))

BATCH_SIZE = 10
TOTAL_QUERIES = 30
MAX_RETRIES = 2

schema_text = Path(SCHEMA_FILE).read_text().strip()

# ------------------------ LOAD MODEL ------------------------

tokenizer, model = load_model_and_tokenizer(
    model_name=MODEL_NAME,
    cache_dir=MODEL_FOLDER,
    device=device,
    use_bnb_8bit=(device == "cuda"),
)

chat_llm = build_chat_llm(
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=MAX_NEW_TOKENS,
)

# ------------------------ GENERATION ------------------------

state = init_state()
all_queries = []

for batch_id in range(TOTAL_QUERIES // BATCH_SIZE):
    avoid_block = render_avoid_block(state)

    prompt = render_prompt(
        schema=schema_text,
        avoid_block=avoid_block,
        batch_size=BATCH_SIZE,
        batch_id=batch_id
    )

    print(f"[info] Generating batch {batch_id + 1}")

    for attempt in range(MAX_RETRIES + 1):
        raw = run_chat_prompt(chat_llm, prompt)

        try:
            batch = coerce_json_array(raw, expected_len=BATCH_SIZE)
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to parse batch {batch_id + 1}\nRaw output:\n{raw}"
                )
            prompt += "\nReturn ONLY a valid JSON array. No text."

    all_queries.extend(batch)
    update_state(state, batch)

# ------------------------ SAVE OUTPUT ------------------------

with open("queries.json", "w") as f:
    json.dump(all_queries, f, indent=2)

print(f"[done] Generated {len(all_queries)} queries")
