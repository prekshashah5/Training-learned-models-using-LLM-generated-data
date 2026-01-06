#!/usr/bin/env python3

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from transformers import logging as hf_logging

from device import get_best_device
from model_loader import load_model_and_tokenizer
from chat_llm import build_chat_llm
from utils import save_output_to_file
from graph import build_generation_graph
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
BATCH_SIZE = 10
TOTAL_QUERIES = 30

initial_state = {
    "schema": schema_text,
    "batch_id": 0,
    "batch_size": BATCH_SIZE,
    "prompt": "",
    "raw_output": "",
    "batch_queries": [],
    "all_queries": [],
}

graph = build_generation_graph(
    chat_llm=chat_llm,
    total_batches=TOTAL_QUERIES // BATCH_SIZE
)


start = time.perf_counter()
final_state = graph.invoke(initial_state)
elapsed = time.perf_counter() - start

all_queries = final_state["all_queries"]

# ------------------------ ASSIGN IDS ------------------------

for i, item in enumerate(all_queries, start=1):
    item["id"] = f"Q{i}"

# ------------------ SAVE OUTPUT ------------------

output_folder = Path(os.getenv("OUTPUT_FOLDER", "../output"))
save_output_to_file(all_queries, output_folder, MODEL_NAME, elapsed, TEMPERATURE)
print(f"[info] Saved {len(all_queries)} queries to {output_folder.resolve()}")