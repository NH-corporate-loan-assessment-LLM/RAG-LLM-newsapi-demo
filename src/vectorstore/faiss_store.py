from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    return vectors / norms


@dataclass
class SearchResult:
    text: str
    metadata: Dict
    score: float


class FaissStore:
    """FAISS-based vector store with simple JSON metadata persistence."""

    def __init__(self, index_dir: str) -> None:
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "store.json")
        self._index: Optional[faiss.Index] = None
        self._entries: List[Dict] = []  # {"id": int, "text": str, "metadata": dict}
        os.makedirs(index_dir, exist_ok=True)
        self._load_if_exists()

    @property
    def size(self) -> int:
        return len(self._entries)

    def _create_index(self, dim: int) -> None:
        self._index = faiss.IndexFlatIP(dim)

    def _load_if_exists(self) -> None:
        if os.path.isfile(self.index_path) and os.path.isfile(self.meta_path):
            self._index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._entries = json.load(f)

    def save(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False)

    def add_texts(self, texts: List[str], embeddings: np.ndarray, metadatas: Optional[List[Dict]] = None) -> List[int]:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(texts) != embeddings.shape[0]:
            raise ValueError("Number of texts and embeddings must match")
        embeddings = embeddings.astype(np.float32)
        embeddings = _normalize(embeddings)
        if self._index is None:
            self._create_index(embeddings.shape[1])
        assert self._index is not None
        self._index.add(embeddings)
        start_id = len(self._entries)
        ids = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            entry_id = start_id + i
            self._entries.append({"id": entry_id, "text": text, "metadata": meta})
            ids.append(entry_id)
        self.save()
        return ids

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        if self._index is None or len(self._entries) == 0:
            return []
        q = query_embedding.astype(np.float32)
        q = _normalize(q.reshape(1, -1))
        scores, indices = self._index.search(q, top_k)
        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            results.append(
                SearchResult(text=entry["text"], metadata=entry["metadata"], score=float(score))
            )
        return results