"""
generate_queries.py
LLM-based query generation using LangGraph and Ollama.
Refactored imports to use the unified package structure.
"""

import math
from langchain_ollama import ChatOllama
from generation.langraph_ollama.prompt import BASE_PROMPT
from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime
import time
from utils.io_utils import save_output_metadata, load_queries_from_temp, append_queries_to_temp, read_json_file
from utils.session_utils import get_latest_json_path
from utils.sql_utils import coerce_json_array, extract_json_array as utils_extract_json_array
from utils.logger import logger
from config.db_config import load_column_stats
import re
import json
import concurrent.futures


def get_generator_llm(model_name: str, temperature: float, BASEURL: str):
    return ChatOllama(
        model=model_name,
        base_url=BASEURL,
        temperature=temperature,
        model_kwargs={"format": "json"},
    )


def build_recent_queries_text(all_queries, limit=15):
    if not all_queries:
        return "None"
    recent = all_queries[-limit:]
    lines = []
    for i, q in enumerate(recent, 1):
        lines.append(f"{i}. {q['sql']}")
    return "\n".join(lines)


def parse_llm_json(raw):
    try:
        return coerce_json_array(raw)
    except Exception:
        return utils_extract_json_array(raw)


def invoke_llm(all_queries, batch_size, schema, llm, model_name: str, max_retries=3):
    stats_text = load_column_stats()

    prompt = BASE_PROMPT.format(
        batch_size=batch_size,
        SCHEMA=schema,
        STATS=stats_text,
    )

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            logger.info(f"[{model_name}] Attempt {attempt+1}/{max_retries}")

            raw_text = ""
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(llm.invoke, prompt)
            try:
                response = future.result(timeout=300)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            raw_text = response.content

            llm_meta = getattr(response, "response_metadata", {}) or {}
            native_duration_ns = llm_meta.get("total_duration")
            if native_duration_ns:
                elapsed_time = native_duration_ns / 1_000_000_000
            else:
                elapsed_time = time.perf_counter() - start_time

            logger.info(f"[{model_name}] LLM invocation took {elapsed_time:.2f} seconds.")

            parsed_json = parse_llm_json(raw_text)
            return True, elapsed_time, parsed_json
        except concurrent.futures.TimeoutError:
            logger.warning(f"[{model_name}] LLM invocation exceeded 5 minute timeout on attempt {attempt+1}")
            if attempt == max_retries - 1:
                return False, None, []
        except Exception as e:
            logger.warning(f"[{model_name}] Error parsing LLM JSON on attempt {attempt+1}: {e}")
            logger.debug(f"[{model_name}] Raw output snippet that failed: {raw_text[:500]}...")
            if attempt == max_retries - 1:
                import traceback
                logger.error("[%s] Final error generating SQL: %s\n%s", model_name, str(e), traceback.format_exc())
                return False, None, []
            time.sleep(1)


REQUIRED_KEYS = {"sql", "type", "reasoning"}


def generate_queries_in_batches(
    total_queries: int,
    batch_size: int,
    schema,
    llm,
    model_name: str,
    temp_file=None,
):
    all_queries = []
    total_elapsed_time = 0.0

    consecutive_failures = 0
    max_consecutive_failures = 5
    batch_num = 1

    while len(all_queries) < total_queries:
        current_batch_size = min(batch_size, total_queries - len(all_queries))

        logger.info(f"[{model_name}] Generating batch {batch_num} (Need {total_queries - len(all_queries)} more queries)")

        success, batch_elapsed_time, result = invoke_llm(
            all_queries=all_queries,
            batch_size=current_batch_size,
            schema=schema,
            llm=llm,
            model_name=model_name,
        )

        if not success:
            consecutive_failures += 1
            logger.warning(f"[{model_name}] Batch {batch_num} failed, skipping. (Consecutive failures: {consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"[{model_name}] Reached {max_consecutive_failures} consecutive failures. Stopping model generation.")
                break
            batch_num += 1
            continue

        if success and batch_elapsed_time is not None:
            total_elapsed_time += batch_elapsed_time

        if isinstance(result, str):
            batch = utils_extract_json_array(result)
        elif isinstance(result, list):
            batch = result
        elif isinstance(result, dict):
            batch = result.get("queries") or result.get("sql_queries") or []
        else:
            consecutive_failures += 1
            logger.warning(f"[{model_name}] Unexpected result type in batch {batch_num}")
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"[{model_name}] Reached {max_consecutive_failures} consecutive failures. Stopping model generation.")
                break
            batch_num += 1
            continue

        clean_batch = []
        for item in batch:
            if not isinstance(item, dict):
                continue
            if not REQUIRED_KEYS.issubset(item.keys()):
                continue
            clean_batch.append(item)

        if not clean_batch:
            consecutive_failures += 1
            logger.warning(f"[{model_name}] Extracted 0 valid queries in batch {batch_num}. (Consecutive failures: {consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"[{model_name}] Reached {max_consecutive_failures} consecutive failures with 0 queries. Stopping model generation.")
                break
            batch_num += 1
            continue

        consecutive_failures = 0

        if temp_file is not None and clean_batch:
            append_queries_to_temp(clean_batch, temp_file)

        logger.info(f"[{model_name}] Extracted {len(clean_batch)} queries in batch {batch_num}. Total so far: {len(all_queries) + len(clean_batch)}")
        all_queries.extend(clean_batch)
        batch_num += 1

    all_queries = all_queries[:total_queries]
    return all_queries, round(total_elapsed_time, 2)


def run_generation():
    MODELS = str(os.getenv("MODELS", "").strip())
    print(f"DEBUG: MODELS env var = '{MODELS}'")
    SCHEMA_FILE = Path(os.getenv("SCHEMA_FILE", "schema.txt"))

    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
    TOTAL_QUERIES = int(os.getenv("TOTAL_QUERIES", 30))

    OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", "../output"))
    TEMP_FOLDER = OUTPUT_FOLDER / "temp"
    TEMP_FOLDER.mkdir(parents=True, exist_ok=True)

    BASE_URL = str(os.getenv("OLLAMA_URL", "http://localhost:11434"))

    models = [m.strip() for m in MODELS.replace(",", " ").split() if m.strip()]

    if not models:
        logger.error("No models found in MODELS variable")

    logger.info(f"Found {len(models)} model(s) to process:")
    for i, model in enumerate(models, 1):
        logger.info(f"  {i}. {model}")

    schema_text = SCHEMA_FILE.read_text().strip()

    results = []
    overall_start = time.perf_counter()

    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = OUTPUT_FOLDER / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created session folder: {session_dir}")

    import threading

    excel_lock = threading.Lock()

    def process_model(idx, model_name):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Model {idx}/{len(models)}: {model_name}")
        logger.info(f"{'#'*70}")

        llm = get_generator_llm(
            model_name=model_name,
            temperature=TEMPERATURE,
            BASEURL=BASE_URL,
        )
        safe_model = model_name.replace("/", "_").replace(" ", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{safe_model}_run_{timestamp}"
        run_dir = session_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        final_file = run_dir / "queries.jsonl"

        (run_dir / "prompt.txt").write_text(BASE_PROMPT, encoding="utf-8")

        all_queries, elapsed = generate_queries_in_batches(
            total_queries=TOTAL_QUERIES,
            batch_size=BATCH_SIZE,
            schema=schema_text,
            llm=llm,
            model_name=model_name,
            temp_file=final_file,
        )

        all_queries = all_queries[:TOTAL_QUERIES]

        with excel_lock:
            save_output_metadata(
                file_path=session_dir,
                MODEL_NAME=model_name,
                run_id=run_id,
                num_queries=len(all_queries),
                elapsed=elapsed,
                TEMPERATURE=TEMPERATURE,
                run_dir=run_dir,
                queries_path=final_file,
            )

        return all_queries

    all_queries_flat = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_model, idx, model_name) for idx, model_name in enumerate(models, 1)]
        for future in concurrent.futures.as_completed(futures):
            try:
                model_queries = future.result()
                all_queries_flat.extend(model_queries)
            except Exception as exc:
                logger.error(f"A model generation failed with an exception: {exc}")

    overall_elapsed = time.perf_counter() - overall_start
    logger.info(f"\nAll models processed in {overall_elapsed:.2f} seconds.")
    latest_json_path = get_latest_json_path(session_dir)
    return all_queries_flat, latest_json_path, session_dir


def get_schema():
    SCHEMA_FILE = Path(os.getenv("SCHEMA_FILE", "schema.txt"))
    schema_text = SCHEMA_FILE.read_text().strip()
    return schema_text
