import os
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv


@dataclass
class Settings:
    # API Keys
    openrouter_api_key: str
    newsapi_key: str
    
    # Vector Store
    index_dir: str
    
    # News Sources (credible sources for corporate analysis)
    news_sources: List[str]
    news_domains: List[str]
    
    # Company Analysis Settings
    default_markets: List[str]
    default_industries: List[str]
    
    # RAG Settings
    chunk_size: int
    chunk_overlap: int
    top_k_results: int
    
    # LLM-based Analysis (no keyword dependency)
    # Keywords removed - LLM will analyze context directly


def _get_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_settings() -> Settings:
    # Load .env if present
    load_dotenv(override=False)

    return Settings(
        # API Keys
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        newsapi_key=os.getenv("NEWSAPI_KEY", ""),
        
        # Vector Store
        index_dir=os.getenv("INDEX_DIR", ".faiss_index"),
        
        # News Sources (credible financial/business sources)
        news_sources=_get_list("NEWS_SOURCES", 
            "bloomberg,reuters,financial-times,wall-street-journal,cnbc,marketwatch,bbc-news,associated-press,forbes,business-insider,yahoo-finance"),
        news_domains=_get_list("NEWS_DOMAINS",
            "bloomberg.com,reuters.com,ft.com,wsj.com,cnbc.com,marketwatch.com,bbc.com,ap.org,forbes.com,businessinsider.com,finance.yahoo.com"),
        
        # Company Analysis
        default_markets=_get_list("DEFAULT_MARKETS", 
            "US,China,EU,Japan,South Korea,India"),
        default_industries=_get_list("DEFAULT_INDUSTRIES",
            "semiconductor,automotive,pharmaceutical,finance,energy,technology"),
        
        # RAG Settings
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k_results=int(os.getenv("TOP_K_RESULTS", "10")),
        
        # LLM-based Analysis (no keyword dependency)
        # Keywords removed - LLM will analyze context directly
    )


def validate_settings(cfg: Settings) -> None:
    missing = []
    if not cfg.openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    if not cfg.newsapi_key:
        missing.append("NEWSAPI_KEY")
    
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
