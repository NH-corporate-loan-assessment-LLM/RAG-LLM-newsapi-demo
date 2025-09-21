from __future__ import annotations

from typing import Iterable, List

import numpy as np
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential


class GeminiEmbedder:
    """Wrapper around Gemini embeddings for documents and queries - following Discord Chatbot pattern"""

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        genai.configure(api_key=api_key)
        self.model = model

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _embed_one(self, text: str, task_type: str) -> List[float]:
        result = genai.embed_content(model=self.model, content=text, task_type=task_type)
        return result["embedding"]

    def embed_texts(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        batch: List[str] = []
        all_vecs: List[List[float]] = []
        for t in texts:
            batch.append(t)
            if len(batch) >= batch_size:
                for item in batch:
                    all_vecs.append(self._embed_one(item, task_type="retrieval_document"))
                batch = []
        if batch:
            for item in batch:
                all_vecs.append(self._embed_one(item, task_type="retrieval_document"))
        return np.array(all_vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self._embed_one(query, task_type="retrieval_query")
        return np.array(vec, dtype=np.float32)
