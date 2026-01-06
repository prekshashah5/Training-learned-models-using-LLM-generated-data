from typing import List, Dict, Any, TypedDict

class GenerationState(TypedDict):
    schema: str
    batch_id: int
    batch_size: int
    prompt: str
    raw_output: str
    batch_queries: List[Dict[str, Any]]
    all_queries: List[Dict[str, Any]]