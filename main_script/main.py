#!/usr/bin/env python3

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from transformers import logging as hf_logging

from device import get_best_device
from model_loader import load_model_and_tokenizer
from chat_llm import build_chat_llm, run_chat_prompt
from prompt import render_prompt
from utils import extract_json_array, save_output_to_file
# ------------------------ SETUP ------------------------

load_dotenv()
device = get_best_device()
print(f"[info] Using device: {device}")

MODEL_NAME = str(os.getenv("MODEL_NAME", "").strip())
MODEL_FOLDER = Path(os.getenv("MODEL_FOLDER", "../models"))
EXCEL_FILE = Path(os.getenv("OUTPUT_FILE", "runs.xlsx"))
SCHEMA_FILE = Path(os.getenv("SCHEMA_FILE", "schema.txt"))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
TOTAL_QUERIES = int(os.getenv("TOTAL_QUERIES", 30))

hf_logging.set_verbosity_info()
hf_logging.enable_explicit_format()

schema_text = SCHEMA_FILE.read_text().strip()

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
    temperature=TEMPERATURE
)

# ------------------ GENERATION LOOP ------------------

all_queries = []

start = time.perf_counter()

for batch_id in range(TOTAL_QUERIES // BATCH_SIZE):

    prompt = render_prompt(
        schema=schema_text,
        batch_size=BATCH_SIZE,
        batch_id=batch_id
    )

    print(f"[info] Generating batch {batch_id + 1}")
    raw = run_chat_prompt(chat_llm, prompt)

    # batch = coerce_json_array(raw, expected_len=BATCH_SIZE)
    batch = extract_json_array(raw)
    print(f"[info] Extracted {len(batch)} queries in batch {batch_id +  1}")
    all_queries.extend(batch)
elapsed = time.perf_counter() - start

i = 1
for item in all_queries:
    item["id"] = f"Q{i}"
    i += 1

# ------------------ SAVE OUTPUT ------------------

output_folder = Path(os.getenv("OUTPUT_FOLDER", "../output"))
save_output_to_file(all_queries, output_folder, MODEL_NAME, elapsed, TEMPERATURE)
print(f"[info] Saved {len(all_queries)} queries to {output_folder.resolve()}")