from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors for cosine similarity"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    return vectors / norms


@dataclass
class SearchResult:
    text: str
    metadata: Dict
    score: float


class FaissStore:
    """FAISS-based vector store with JSON metadata persistence for news articles"""

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
        """Create FAISS index for given dimension"""
        self._index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine Similarity

    def _load_if_exists(self) -> None:
        """Load existing index and metadata if available"""
        if os.path.isfile(self.index_path) and os.path.isfile(self.meta_path):
            try:
                self._index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                print(f"Loaded existing index with {len(self._entries)} entries")
            except Exception as e:
                print(f"Error loading existing index: {e}")
                self._index = None
                self._entries = []

    def save(self) -> None:
        """Save index and metadata to disk"""
        if self._index is None:
            return
        try:
            faiss.write_index(self._index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
            print(f"Saved index with {len(self._entries)} entries")
        except Exception as e:
            print(f"Error saving index: {e}")

    def add_texts(
        self, 
        texts: List[str], 
        embeddings: np.ndarray, 
        metadatas: Optional[List[Dict]] = None
    ) -> List[int]:
        """Add texts with embeddings to the store"""
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
            self._entries.append({
                "id": entry_id, 
                "text": text, 
                "metadata": meta
            })
            ids.append(entry_id)
        
        self.save()
        return ids

    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar texts"""
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
                SearchResult(
                    text=entry["text"], 
                    metadata=entry["metadata"], 
                    score=float(score)
                )
            )
        
        return results

    def search_by_metadata(
        self,
        query_embedding: np.ndarray,
        metadata_filter: Dict[str, Any],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search with metadata filtering"""
        if self._index is None or len(self._entries) == 0:
            return []
        
        # First get more results than needed
        q = query_embedding.astype(np.float32)
        q = _normalize(q.reshape(1, -1))
        
        scores, indices = self._index.search(q, min(top_k * 3, len(self._entries)))
        
        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue
            
            entry = self._entries[idx]
            metadata = entry["metadata"]
            
            # Apply metadata filter
            if self._matches_filter(metadata, metadata_filter):
                results.append(
                    SearchResult(
                        text=entry["text"],
                        metadata=metadata,
                        score=float(score)
                    )
                )
                
                if len(results) >= top_k:
                    break
        
        return results

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        if not self._entries:
            return {"total_entries": 0}
        
        # Count by source
        source_counts = {}
        for entry in self._entries:
            source = entry["metadata"].get("source_name", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_entries": len(self._entries),
            "sources": source_counts,
            "index_dimension": self._index.d if self._index else 0,
        }

