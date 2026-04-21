import numpy as np
from typing import List, Dict, Any, Tuple


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class InMemoryVectorDB:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.documents: List[Dict[str, Any]] = []

    def add(self, document: Dict[str, Any], embedding: np.ndarray) -> None:
        self.documents.append(document)
        self.embeddings.append(embedding)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        if len(self.embeddings) == 0:
            return []

        similarities = [cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        return [(self.documents[i], similarities[i]) for i in ranked_indices[:top_k]]

    def build_from_documents(self, documents: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> None:
        self.documents = documents
        self.embeddings = embeddings
