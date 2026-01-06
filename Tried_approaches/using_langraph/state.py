from typing import TypedDict, List, Dict, Set, Any

class GraphState(TypedDict):
    schema: str
    batch_size: int
    total_batches: int
    batch_id: int

    avoid: Dict[str, Set[str]]
    queries: List[Dict[str, Any]]

    prompt: str
    raw: str
    parsed: List[Dict[str, Any]]
    retries: int
