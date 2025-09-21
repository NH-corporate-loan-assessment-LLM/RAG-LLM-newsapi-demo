from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import requests
from newsapi import NewsApiClient


@dataclass
class NewsArticle:
    """News article data structure"""
    
    title: str
    description: str
    content: str
    url: str
    published_at: datetime
    source: str
    source_domain: str
    
    # Additional metadata
    author: Optional[str] = None
    url_to_image: Optional[str] = None
    language: str = "en"
    
    def to_text(self) -> str:
        """Convert article to searchable text"""
        content_text = self.content or self.description or ""
        return f"[{self.source}] {self.title}\n{content_text}\nSource: {self.url}"
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dictionary"""
        return {
            "source": self.url,
            "title": self.title,
            "source_name": self.source,
            "source_domain": self.source_domain,
            "author": self.author,
            "published_at": self.published_at.isoformat(),
            "url_to_image": self.url_to_image,
            "language": self.language,
        }


class NewsApiIngestor:
    """NewsAPI client for fetching credible business news"""
    
    def __init__(self, api_key: str, sources: List[str], domains: List[str]):
        self.api = NewsApiClient(api_key=api_key)
        self.sources = sources
        self.domains = domains
    
    def fetch_articles(
        self,
        query: str,
        days_back: int = 7,  # Reduced to 7 days for free plan
        page_size: int = 100,
        language: str = "en",
        sort_by: str = "publishedAt"
    ) -> List[NewsArticle]:
        """Fetch articles based on query"""
        
        # Calculate date range (limit to 3 days for free plan)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=min(days_back, 3))
        
        articles = []
        page = 1
        
        while len(articles) < page_size and page <= 5:  # Max 5 pages
            try:
                # Search with sources
                response = self.api.get_everything(
                    q=query,
                    sources=",".join(self.sources) if self.sources else None,
                    domains=",".join(self.domains) if self.domains else None,
                    from_param=from_date.strftime("%Y-%m-%d"),
                    to=to_date.strftime("%Y-%m-%d"),
                    language=language,
                    sort_by=sort_by,
                    page=page,
                    page_size=min(20, page_size - len(articles))  # API limit
                )
                
                if response['status'] != 'ok':
                    error_msg = response.get('message', 'Unknown error')
                    print(f"NewsAPI error: {error_msg}")
                    # If it's a date range error, try without date filter
                    if 'too far in the past' in error_msg.lower():
                        print("Retrying without date filter...")
                        response = self.api.get_everything(
                            q=query,
                            sources=",".join(self.sources) if self.sources else None,
                            domains=",".join(self.domains) if self.domains else None,
                            language=language,
                            sort_by=sort_by,
                            page=page,
                            page_size=min(20, page_size - len(articles))
                        )
                        if response['status'] != 'ok':
                            print(f"NewsAPI error (retry): {response.get('message', 'Unknown error')}")
                            break
                    else:
                        break
                
                for article_data in response.get('articles', []):
                    if len(articles) >= page_size:
                        break
                    
                    article = self._parse_article(article_data)
                    if article:
                        articles.append(article)
                
                if not response.get('articles'):
                    break
                    
                page += 1
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching articles: {e}")
                break
        
        return articles
    
    def fetch_company_news(
        self,
        company_profile,
        days_back: int = 30,
        max_articles: int = 200
    ) -> List[NewsArticle]:
        """Fetch news relevant to a company profile"""
        
        all_articles = []
        search_terms = company_profile.get_search_keywords()
        
        # Search for each term
        for term in search_terms[:10]:  # Limit to avoid rate limits
            articles = self.fetch_articles(
                query=term,
                days_back=days_back,
                page_size=50
            )
            all_articles.extend(articles)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles[:max_articles]
    
    def fetch_policy_news(
        self,
        company_profile,
        days_back: int = 90,  # Longer period for policy changes
        max_articles: int = 100
    ) -> List[NewsArticle]:
        """Fetch policy and regulation news relevant to company"""
        
        all_articles = []
        policy_terms = company_profile.get_policy_search_terms()
        
        for term in policy_terms[:15]:  # Limit search terms
            articles = self.fetch_articles(
                query=term,
                days_back=days_back,
                page_size=30
            )
            all_articles.extend(articles)
        
        # Remove duplicates
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles[:max_articles]
    
    def fetch_market_trends(
        self,
        company_profile,
        days_back: int = 60,
        max_articles: int = 100
    ) -> List[NewsArticle]:
        """Fetch market trend news relevant to company"""
        
        all_articles = []
        trend_terms = company_profile.get_market_trend_terms()
        
        for term in trend_terms[:15]:
            articles = self.fetch_articles(
                query=term,
                days_back=days_back,
                page_size=30
            )
            all_articles.extend(articles)
        
        # Remove duplicates
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles[:max_articles]
    
    def _parse_article(self, article_data: Dict[str, Any]) -> Optional[NewsArticle]:
        """Parse article data from NewsAPI response"""
        try:
            # Parse published date
            published_str = article_data.get('publishedAt', '')
            if published_str:
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            else:
                published_at = datetime.now()
            
            return NewsArticle(
                title=article_data.get('title', ''),
                description=article_data.get('description', ''),
                content=article_data.get('content', ''),
                url=article_data.get('url', ''),
                published_at=published_at,
                source=article_data.get('source', {}).get('name', 'Unknown'),
                source_domain=article_data.get('url', '').split('/')[2] if article_data.get('url') else 'Unknown',
                author=article_data.get('author'),
                url_to_image=article_data.get('urlToImage'),
            )
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None

