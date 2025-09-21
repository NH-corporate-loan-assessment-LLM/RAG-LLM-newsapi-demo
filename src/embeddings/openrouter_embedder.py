from __future__ import annotations

from typing import Iterable, List
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenRouterEmbedder:
    """OpenRouter embeddings wrapper for documents and queries"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large") -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _embed_one(self, text: str) -> List[float]:
        """Embed a single text with retry logic"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "input": text
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error embedding text: {e}")
            raise

    def embed_texts(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        """Embed multiple texts for document storage (Discord Chatbot style)"""
        batch: List[str] = []
        all_vecs: List[List[float]] = []
        
        for t in texts:
            batch.append(t)
            if len(batch) >= batch_size:
                for item in batch:
                    all_vecs.append(self._embed_one(item))
                batch = []
        
        if batch:
            for item in batch:
                all_vecs.append(self._embed_one(item))
        
        return np.array(all_vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query for search"""
        vec = self._embed_one(query)
        return np.array(vec, dtype=np.float32)

    def embed_company_context(self, company_profile) -> np.ndarray:
        """Embed company context for specialized search"""
        context_text = f"""
        Company: {company_profile.company_name}
        Industry: {company_profile.industry}
        Products/Services: {', '.join(company_profile.main_products_services)}
        Target Markets: {', '.join(company_profile.target_markets)}
        Business Model: {company_profile.business_model or 'Not specified'}
        """
        return self.embed_query(context_text)

