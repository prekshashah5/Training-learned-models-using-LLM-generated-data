# main.py
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
from llm import build_llm
from graph import build_graph
from dotenv import load_dotenv
import os
from pathlib import Path

from device import get_best_device
from model_loader import load_model_and_tokenizer

load_dotenv()
hf_logging.set_verbosity_info()
hf_logging.enable_explicit_format()
MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
MODEL_FOLDER = Path(os.getenv("MODEL_FOLDER", "../models"))
SCHEMA_FILE = Path(os.getenv("SCHEMA_FILE", "schema.txt"))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))

BATCH_SIZE = 10
TOTAL_QUERIES = 1
MAX_RETRIES = 2

schema_text = SCHEMA_FILE.read_text().strip()
device = get_best_device()


tokenizer, model = load_model_and_tokenizer(
    model_name=MODEL_NAME,
    cache_dir=MODEL_FOLDER,
    device=device,
    use_bnb_8bit=(device == "cuda"),
)

llm = build_llm(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    max_new_tokens=512
)

state = {
    "schema": schema_text,
    "batch_size": BATCH_SIZE,
    "total_batches": TOTAL_QUERIES // BATCH_SIZE,
    "batch_id": 1,

    "avoid": {
        "types": set(),
        "columns": set(),
        "predicates": set(),
    },

    "queries": [],
    "prompt": "",
    "raw": "",
    "parsed": [],
    "retries": 0,
}

app = build_graph(llm)
print(app.get_graph().draw_mermaid())

png_bytes = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)

final_state = app.invoke(state)

with open("queries.json", "w") as f:
    json.dump(final_state["queries"], f, indent=2)

print("Done:", len(final_state["queries"]))
