# graph.py
from langgraph.graph import StateGraph, END
from state import GenerationState
from nodes import (
    build_prompt_node,
    extract_json_node,
    accumulate_node,
    llm_generate_node,
)

def build_generation_graph(chat_llm, total_batches: int):

    graph = StateGraph(GenerationState)

    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("llm_generate", llm_generate_node(chat_llm))
    graph.add_node("extract_json", extract_json_node)
    graph.add_node("accumulate", accumulate_node)

    graph.set_entry_point("build_prompt")

    graph.add_edge("build_prompt", "llm_generate")
    graph.add_edge("llm_generate", "extract_json")
    graph.add_edge("extract_json", "accumulate")

    def should_continue(state):
        if state["batch_id"] >= total_batches:
            return END
        return "build_prompt"

    graph.add_conditional_edges(
        "accumulate",
        should_continue,
    )

    return graph.compile()
