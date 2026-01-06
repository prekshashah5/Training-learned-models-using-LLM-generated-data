# graph.py
from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import (
    prompt_node,
    llm_node,
    parse_node,
    retry_router,
    update_node,
    continue_router,
)

def build_graph(llm):
    graph = StateGraph(GraphState)

    graph.add_node("prompt", prompt_node)
    graph.add_node("llm", lambda s: llm_node(s, llm))
    graph.add_node("parse", parse_node)
    graph.add_node("update", update_node)

    graph.set_entry_point("prompt")

    graph.add_edge("prompt", "llm")
    graph.add_edge("llm", "parse")

    graph.add_conditional_edges(
        "parse",
        retry_router,
        {"llm": "llm", "update": "update"}
    )

    graph.add_conditional_edges(
        "update",
        continue_router,
        {"prompt": "prompt", "end": END}
    )

    return graph.compile()
