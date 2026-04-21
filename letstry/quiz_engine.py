import json
import os
from typing import Any, Dict, List

import numpy as np

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "plan_details.json")


def load_knowledge_base(path: str = DATA_FILE) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def text_to_embedding(text: str) -> np.ndarray:
    normalized = text.lower().strip()
    vector = np.zeros(300, dtype=float)
    for index, token in enumerate(normalized.split(), start=1):
        token_value = sum(ord(ch) for ch in token)
        vector[index % 300] += token_value
    if np.linalg.norm(vector) == 0:
        return vector
    return vector / np.linalg.norm(vector)


def build_vector_db(plan_contents: List[Dict[str, Any]]) -> "InMemoryVectorDB":
    from vector_db import InMemoryVectorDB

    store = InMemoryVectorDB()
    for plan in plan_contents:
        content = plan.get("content", "")
        title = plan.get("title", "")
        metadata = {
            "id": plan.get("id"),
            "title": title,
            "content": content,
            "keywords": plan.get("keywords", []),
        }
        embedding = text_to_embedding(f"{title}. {content}")
        store.add(metadata, embedding)
    return store


def retrieve_context(store: "InMemoryVectorDB", question: str, top_k: int = 2) -> List[Dict[str, Any]]:
    query_embedding = text_to_embedding(question)
    results = store.search(query_embedding, top_k=top_k)
    return [result[0] for result in results]


def build_context_text(context_docs: List[Dict[str, Any]]) -> str:
    segments = []
    for doc in context_docs:
        segments.append(f"Title: {doc['title']}\nContent: {doc['content']}")
    return "\n\n".join(segments)
