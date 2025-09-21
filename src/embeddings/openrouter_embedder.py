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
        
        # Validate API key format
        if not self.api_key or not self.api_key.startswith('sk-'):
            print(f"Warning: API key format may be incorrect. Expected 'sk-...' format, got: {self.api_key[:10]}...")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _embed_one(self, text: str) -> List[float]:
        """Embed a single text with retry logic using OpenAI-compatible API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",  # Required by OpenRouter
                "X-Title": "News Analysis App"  # Required by OpenRouter
            }
            
            data = {
                "model": self.model,
                "input": text
            }
            
            print(f"Making embedding request to OpenRouter...")
            print(f"API Key: {self.api_key[:10]}...")
            print(f"Model: {self.model}")
            print(f"URL: {self.base_url}/embeddings")
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            
            # Check if response is HTML (indicates wrong endpoint or auth issue)
            if response.text.strip().startswith('<!DOCTYPE html>') or response.text.strip().startswith('<html'):
                print(f"Received HTML response. This might be due to:")
                print(f"1. Invalid API key")
                print(f"2. Wrong endpoint")
                print(f"3. Authentication issue")
                print(f"Response preview: {response.text[:200]}...")
                
                # Try with a fallback approach - use a different model
                print("Trying with a different model...")
                fallback_data = {
                    "model": "text-embedding-ada-002",  # Fallback model
                    "input": text
                }
                
                fallback_response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=fallback_data,
                    timeout=30
                )
                
                if fallback_response.text.strip().startswith('<!DOCTYPE html>'):
                    raise Exception("Both primary and fallback models failed. Check API key and endpoint.")
                
                response = fallback_response
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Check if the response has the expected structure
                    if "data" in result and len(result["data"]) > 0:
                        if "embedding" in result["data"][0]:
                            return result["data"][0]["embedding"]
                        else:
                            print(f"Data structure: {result['data'][0].keys()}")
                            raise Exception(f"Unexpected data structure: {result['data'][0]}")
                    else:
                        raise Exception(f"Unexpected response structure: {result}")
                except Exception as json_error:
                    raise Exception(f"Failed to parse JSON response: {json_error}\nResponse: {response.text[:500]}")
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

