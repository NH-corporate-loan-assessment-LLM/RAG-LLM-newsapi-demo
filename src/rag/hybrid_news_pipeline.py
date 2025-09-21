from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from datetime import datetime

from src.news.newsapi_client import NewsArticle, NewsApiIngestor
from src.embeddings.gemini_embedder import GeminiEmbedder
from src.vectorstore.faiss_store import FaissStore, SearchResult
from src.analysis.openrouter_impact_scorer import OpenRouterImpactScorer


class HybridNewsRagPipeline:
    """Hybrid News RAG Pipeline: Gemini embeddings + OpenRouter LLM"""
    
    def __init__(self, google_api_key: str, openrouter_api_key: str, newsapi_key: str, index_dir: str):
        # Gemini for embeddings (stable)
        self.embedder = GeminiEmbedder(api_key=google_api_key)
        self.store = FaissStore(index_dir=index_dir)
        
        # News API client
        self.news_client = NewsApiIngestor(
            api_key=newsapi_key,
            sources=["reuters", "bloomberg", "financial-times", "cnbc"],
            domains=["reuters.com", "bloomberg.com", "ft.com", "cnbc.com"]
        )
        
        # OpenRouter for LLM analysis (cost-effective)
        self.impact_scorer = OpenRouterImpactScorer(
            openrouter_api_key=openrouter_api_key,
            model="gpt-4o-mini"
        )
    
    def fetch_and_store_articles(self, query: str, max_articles: int = 50) -> int:
        """Fetch articles from News API and store them in vector database"""
        print(f"Fetching articles for query: {query}")
        
        # Step 1: Fetch articles using News API
        articles = self.news_client.fetch_articles(
            query=query,
            days_back=7,
            page_size=max_articles
        )
        
        print(f"Fetched {len(articles)} articles")
        
        if not articles:
            print("No articles found")
            return 0
        
        # Step 2: Convert articles to text chunks and metadata
        documents = []
        for article in articles:
            # Create searchable text
            text = f"[{article.source}] {article.title}\n{article.description or ''}\n{article.content or ''}"
            
            # Create metadata
            metadata = {
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "published_at": article.published_at.isoformat(),
                "author": article.author,
                "query": query  # Store the original query
            }
            
            documents.append((text, metadata))
        
        # Step 3: Embed using Gemini and store in vector database
        texts = [doc[0] for doc in documents]
        metadatas = [doc[1] for doc in documents]
        
        print("Creating embeddings with Gemini...")
        vectors = self.embedder.embed_texts(texts)
        
        print("Storing in vector database...")
        self.store.add_texts(texts, vectors, metadatas)
        
        print(f"Successfully stored {len(documents)} articles")
        return len(documents)
    
    def search_articles(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for relevant articles using vector similarity"""
        print(f"Searching for articles with query: {query}")
        
        # Create query embedding using Gemini
        query_vector = self.embedder.embed_query(query)
        
        # Search vector database
        results = self.store.search(query_vector, top_k=top_k)
        
        print(f"Found {len(results)} relevant articles")
        return results
    
    def find_best_article(self, company_context: str, query: str = None) -> Optional[Dict[str, Any]]:
        """Find the single best article using similarity scoring"""
        
        # Use company context as search query if no specific query provided
        search_query = query or company_context
        
        # Search for relevant articles
        results = self.search_articles(search_query, top_k=20)
        
        if not results:
            print("No articles found")
            return None
        
        print(f"Analyzing {len(results)} articles for best match...")
        
        best_article = None
        best_score = -1
        
        # Score each article based on similarity and relevance
        for result in results:
            # Combine similarity score with additional factors
            similarity_score = result.score
            
            # Additional scoring factors
            metadata = result.metadata
            
            # Boost score for recent articles
            try:
                published_at = datetime.fromisoformat(metadata.get("published_at", ""))
                days_old = (datetime.now() - published_at.replace(tzinfo=None)).days
                recency_boost = max(0, 1 - (days_old / 30))  # Decay over 30 days
            except:
                recency_boost = 0.5  # Default for unknown dates
            
            # Boost score for credible sources
            source = metadata.get("source", "").lower()
            source_boost = 1.0
            if any(credible in source for credible in ["reuters", "bloomberg", "financial times"]):
                source_boost = 1.2
            elif any(credible in source for credible in ["cnbc", "bbc", "forbes"]):
                source_boost = 1.1
            
            # Calculate final score
            final_score = similarity_score * recency_boost * source_boost
            
            if final_score > best_score:
                best_score = final_score
                best_article = {
                    "article": result,
                    "score": final_score,
                    "similarity": similarity_score,
                    "recency_boost": recency_boost,
                    "source_boost": source_boost,
                    "metadata": metadata
                }
        
        if best_article:
            print(f"Best article found: {best_article['metadata']['title']}")
            print(f"Score: {best_article['score']:.3f} (similarity: {best_article['similarity']:.3f})")
            print(f"Source: {best_article['metadata']['source']}")
        
        return best_article
    
    def analyze_article_impact(self, article_data: Dict[str, Any], company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of the selected article using OpenRouter LLM"""
        
        if not article_data:
            return {"error": "No article provided"}
        
        article = article_data["article"]
        article_text = article.text
        article_metadata = article_data["metadata"]
        
        # Create a mock company profile object for the impact scorer
        class MockCompanyProfile:
            def __init__(self, profile_dict):
                self.name = profile_dict.get("name", "Unknown Company")
                self.industry = profile_dict.get("industry", "Unknown")
                self.products_services = profile_dict.get("products_services", [])
                self.target_markets = profile_dict.get("target_markets", [])
                self.business_model = profile_dict.get("business_model", "Unknown")
        
        company_obj = MockCompanyProfile(company_profile)
        
        # Analyze impact using OpenRouter LLM
        try:
            print("Analyzing article impact with OpenRouter LLM...")
            impact_score = self.impact_scorer.analyze_article_impact(
                article_text=article_text,
                article_metadata=article_metadata,
                company_profile=company_obj
            )
            
            return {
                "impact_analysis": impact_score,
                "article_score": article_data["score"],
                "article_metadata": article_data["metadata"]
            }
            
        except Exception as e:
            print(f"Error analyzing article impact: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "article_score": article_data["score"],
                "article_metadata": article_data["metadata"]
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        return {
            "vector_store_size": self.store.size,
            "embedder_model": self.embedder.model,
            "llm_model": self.impact_scorer.model,
            "news_sources": self.news_client.sources,
            "embedding_provider": "Gemini",
            "llm_provider": "OpenRouter"
        }
