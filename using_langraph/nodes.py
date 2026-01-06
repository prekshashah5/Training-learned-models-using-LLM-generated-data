# prompt → LLM → raw text → JSON extraction → accumulation → save

from prompt import render_prompt
from chat_llm import run_chat_prompt
from utils import extract_json_array

def build_prompt_node(state):
    prompt = render_prompt(
        schema=state["schema"],
        batch_size=state["batch_size"],
        batch_id=state["batch_id"],
    )
    return {
        **state,
        "prompt": prompt,
    }

def llm_generate_node(chat_llm):
    def _node(state):
        raw = run_chat_prompt(chat_llm, state["prompt"])
        return {
            **state,
            "raw_output": raw,
        }
    return _node

def extract_json_node(state):
    batch = extract_json_array(state["raw_output"])
    return {
        **state,
        "batch_queries": batch,
    }

def accumulate_node(state):
    all_q = state.get("all_queries", [])
    all_q.extend(state["batch_queries"])
    return {
        **state,
        "all_queries": all_q,
        "batch_id": state["batch_id"] + 1,
    }