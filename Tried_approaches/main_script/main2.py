#!/usr/bin/env python3

import os
import time
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from transformers import logging as hf_logging

from device import get_best_device
from model_loader import load_model_and_tokenizer
from excel_utils import append_row_to_excel
from validation import verify_queries_exist, print_verification_report
from chat_llm import build_chat_llm, run_chat_prompt
from prompt import render_prompt
import json
# ------------------------ SETUP ------------------------

load_dotenv()
device = get_best_device()
print(f"[info] Using device: {device}")

MODEL_NAME = str(os.getenv("MODEL_NAME", "").strip())
MODEL_FOLDER = Path(os.getenv("MODEL_FOLDER", "../models"))
PROMPT_FILE = os.getenv("PROMPT_FILE", "prompt.txt")
EXCEL_FILE = Path(os.getenv("OUTPUT_FILE", "runs.xlsx"))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))

hf_logging.set_verbosity_info()
hf_logging.enable_explicit_format()

schema_text = Path(os.getenv("SCHEMA_FILE", "schema.txt")).read_text().strip()
# prompt = replace_schema(schema_text)
# print("[info] Loaded prompt:", prompt)

# ------------------------ LOAD MODEL ------------------------
tokenizer, model = load_model_and_tokenizer(
    model_name=MODEL_NAME,
    cache_dir=MODEL_FOLDER,
    device=device,
    use_bnb_8bit=(device == "cuda")
)

chat_llm = build_chat_llm(
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=MAX_NEW_TOKENS,
)

# ------------------ GENERATION LOOP ------------------

all_queries = []
BATCH_SIZE = 10
TOTAL_QUERIES = 30

for batch_id in range(TOTAL_QUERIES // BATCH_SIZE):

    prompt = render_prompt(
        schema=schema_text,
        batch_size=BATCH_SIZE,
    )

    print(f"[info] Generating batch {batch_id + 1}")
    raw = run_chat_prompt(chat_llm, prompt)

    batch = json.loads(raw)
    all_queries.extend(batch)

# ------------------ SAVE OUTPUT ------------------

with open("queries.json", "w") as f:
    json.dump(all_queries, f, indent=2)

print(f"[done] Generated {len(all_queries)} queries")
# # ------------------------ GENERATION ------------------------

# print("[info] Generating...")
# start = time.perf_counter()

# chat_llm = build_chat_llm(
#     model=model,
#     tokenizer=tokenizer,
#     device=device,
#     max_new_tokens=MAX_NEW_TOKENS,
#     temperature=TEMPERATURE
# )

# generated_text = run_chat_prompt(chat_llm, prompt)

# elapsed = time.perf_counter() - start

# print(generated_text)
# print(f"[info] Generation took {elapsed:.2f} seconds.")

# # ------------------------ LOGGING ------------------------

# append_row_to_excel(
#     EXCEL_FILE,
#     [
#         datetime.now().isoformat(),
#         MODEL_NAME,
#         generated_text,
#         elapsed,
#         prompt
#     ],
#     header=["Timestamp", "Model", "Output", "Time", "Prompt"]
# )

# # ------------------------ VALIDATION ------------------------

# result = verify_queries_exist(generated_text)
# print_verification_report(result)