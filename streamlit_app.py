#!/usr/bin/env python3
"""
Corporate Loan Assessment RAG System - Streamlit Frontend
회사 키워드 + 해외진출 지역 → ML Feature 분석
"""

import streamlit as st
import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_settings, validate_settings
from src.news.newsapi_client import NewsApiIngestor
from src.rag.hybrid_news_pipeline import HybridNewsRagPipeline
from src.vectorstore.faiss_store import FaissStore
from src.chunking.text import chunk_text

# Page config
st.set_page_config(
    page_title="NH 기업대출 심사 2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #000000;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }
    .nh-logo {
        text-align: center;
        margin-bottom: 1rem;
    }
    .nh-logo img {
        max-width: 200px;
        height: auto;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .article-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    .stSelectbox > div > div {
        border-radius: 6px;
    }
    .stTextInput > div > div > input {
        border-radius: 6px;
    }
    
    /* 사이드바 스타일링 */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 3px solid #28a745;
    }
    
    /* 사이드바 헤더 스타일 */
    .css-1d391kg .stHeader {
        color: #2c3e50;
        border-bottom: 1px solid #e9ecef;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* 메인 콘텐츠 영역 중앙 정렬 */
    .main .block-container {
        border-left: 3px solid #28a745;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: calc(100vw - 350px); /* 사이드바 폭(약 350px) 제외 */
        margin: 0 auto; /* 중앙 정렬 */
        margin-left: calc(50vw - 175px); /* 사이드바 폭의 절반만큼 오른쪽으로 이동 */
    }
    
    /* 메인 컨텐츠 전체 영역 중앙 정렬 */
    .main {
        padding-left: 0 !important;
    }
    
    /* 헤더 영역도 중앙 정렬 */
    .main .block-container > div:first-child {
        text-align: center;
    }
    
    /* 사이드바 로고 배경 */
    .css-1d391kg img {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load and cache system components"""
    try:
        settings = load_settings()
        validate_settings(settings)
        
        # Initialize hybrid pipeline
        pipeline = HybridNewsRagPipeline(
            google_api_key=settings.google_api_key,
            openrouter_api_key=settings.openrouter_api_key,
            newsapi_key=settings.newsapi_key,
            index_dir=settings.index_dir
        )
        
        return {
            "settings": settings,
            "pipeline": pipeline
        }
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        return None

def generate_search_queries(company_keywords: str, export_regions: str) -> List[str]:
    """Generate simplified and more effective search queries"""
    keywords = [k.strip() for k in company_keywords.split(',') if k.strip()]
    regions = [r.strip() for r in export_regions.split(',') if r.strip()]
    
    queries = []
    
    # 1. Simple and direct queries (HIGH PRIORITY)
    for keyword in keywords[:2]:  # Top 2 keywords
        # Basic keyword + Korea combinations
        queries.append(f'"{keyword}" Korea')
        queries.append(f'"{keyword}" "South Korea"')
        queries.append(f'Korean "{keyword}"')
        
        # Business and trade focus
        queries.append(f'"{keyword}" Korea business')
        queries.append(f'"{keyword}" Korea trade')
        queries.append(f'"{keyword}" Korea export')
        
        # Market and industry focus
        queries.append(f'"{keyword}" Korea market')
        queries.append(f'"{keyword}" Korea industry')
        queries.append(f'"{keyword}" Korea technology')
    
    # 2. Regional combinations (MEDIUM PRIORITY)
    for region in regions[:3]:  # Top 3 regions
        for keyword in keywords[:1]:  # Top 1 keyword
            queries.append(f'"{keyword}" Korea "{region}"')
            queries.append(f'Korean "{keyword}" "{region}"')
            queries.append(f'"{keyword}" "{region}" trade')
            queries.append(f'"{keyword}" "{region}" market')
    
    # 3. Policy and regulation (LOWER PRIORITY)
    for keyword in keywords[:1]:
        queries.append(f'"{keyword}" Korea policy')
        queries.append(f'"{keyword}" Korea regulation')
        queries.append(f'"{keyword}" Korea government')
        queries.append(f'"{keyword}" Korea tariff')
    
    # 4. Industry-specific terms
    for keyword in keywords[:1]:
        queries.append(f'"{keyword}" Korea semiconductor')
        queries.append(f'"{keyword}" Korea chip')
        queries.append(f'"{keyword}" Korea memory')
        queries.append(f'"{keyword}" Korea processor')
    
    return queries

def calculate_relevance_score(article, company_keywords: str, export_regions: str) -> float:
    """Calculate relevance score for an article based on keywords and regions"""
    keywords = [k.strip().lower() for k in company_keywords.split(',') if k.strip()]
    regions = [r.strip().lower() for r in export_regions.split(',') if r.strip()]
    
    # Combine title and content for analysis
    text = f"{article.title} {article.content}".lower()
    
    score = 0.0
    
    # Keyword matching (higher weight for title)
    title_text = article.title.lower()
    for keyword in keywords:
        if keyword in title_text:
            score += 3.0  # High weight for title matches
        if keyword in text:
            score += 1.0  # Lower weight for content matches
    
    # Region matching
    for region in regions:
        if region in title_text:
            score += 2.0  # High weight for title matches
        if region in text:
            score += 0.5  # Lower weight for content matches
    
    # Korea-related terms (boost for Korean context)
    korea_terms = ['korea', 'korean', 'south korea', 'seoul']
    for term in korea_terms:
        if term in title_text:
            score += 1.5
        if term in text:
            score += 0.3
    
    # Business/trade terms (boost for business relevance)
    business_terms = ['business', 'trade', 'export', 'market', 'industry', 'company', 'corporate']
    for term in business_terms:
        if term in title_text:
            score += 1.0
        if term in text:
            score += 0.2
    
    # Penalize very short articles
    if len(article.content) < 200:
        score *= 0.5
    
    # Penalize articles without Korea context
    if not any(term in text for term in korea_terms):
        score *= 0.3
    
    return score

def fetch_and_analyze_news(system, company_keywords: str, export_regions: str, days_back: int = 30, enable_vector_similarity: bool = True):
    """Fetch news and perform RAG analysis"""
    
    # Generate search queries
    queries = generate_search_queries(company_keywords, export_regions)
    
    # Show search queries for debugging
    with st.expander("Search Queries Used"):
        st.write(f"**Total queries:** {len(queries)}")
        for i, query in enumerate(queries[:10], 1):  # Show first 10 queries
            st.write(f"{i}. {query}")
        if len(queries) > 10:
            st.write(f"... and {len(queries) - 10} more queries")
    
    # Fetch and store articles using the hybrid pipeline
    total_stored = 0
    successful_queries = 0
    for query in queries:
        try:
            num_stored = system["pipeline"].fetch_and_store_articles(
                query=query,
                max_articles=10
            )
            total_stored += num_stored
            if num_stored > 0:
                successful_queries += 1
        except Exception as e:
            st.warning(f"Error searching '{query}': {e}")
    
    # Show search statistics
    st.info(f"Search completed: {successful_queries}/{len(queries)} queries returned results, {total_stored} total articles stored")
    
    # Find the best article using similarity search
    company_context = f"Company: {company_keywords}, Export regions: {export_regions}"
    best_article = system["pipeline"].find_best_article(company_context)
    
    if not best_article:
        st.warning("No relevant articles found. Please try different keywords or check your search terms.")
        return
    
    # Display the best article
    st.success(f"Best matching article found (Score: {best_article['score']:.3f})")
    article_metadata = best_article['metadata']
    
    with st.expander(f"{article_metadata['title']}", expanded=True):
        st.write(f"**Source:** {article_metadata['source']}")
        st.write(f"**Published:** {article_metadata['published_at']}")
        st.write(f"**URL:** {article_metadata['url']}")
        st.write(f"**Similarity Score:** {best_article['similarity']:.3f}")
        st.write(f"**Recency Boost:** {best_article['recency_boost']:.3f}")
        st.write(f"**Source Boost:** {best_article['source_boost']:.3f}")
    
    # For analysis, we'll use the best article
    articles_list = [best_article]
    
    if not articles_list:
        return None, []
    
    # Process articles with RAG
    processed_articles = []
    ml_features_list = []
    
    for article in articles_list[:5]:  # Top 5 articles (reduced to limit API calls)
        try:
            # For hybrid pipeline, article is already a dict with article data
            # Get article text from the article object
            article_obj = article["article"]
            article_text = article_obj.text
            chunks = chunk_text(article_text, 1000, 200)
            
            # Create embeddings for chunks and calculate similarity scores
            chunk_similarities = []
            if enable_vector_similarity:
                try:
                    # Create query embedding for company context (once per article)
                    query_text = f"{company_keywords} {export_regions} business market"
                    query_embedding = system["embedder"].embed_query(query_text)
                    
                    # Limit to first 3 chunks to reduce API calls
                    limited_chunks = chunks[:3]
                    
                    for chunk in limited_chunks:
                        try:
                            # Create embedding for chunk
                            chunk_embedding = system["embedder"].embed_query(chunk)
                            
                            # Calculate cosine similarity
                            similarity = np.dot(chunk_embedding, query_embedding) / (
                                np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embedding)
                            )
                            chunk_similarities.append(similarity)
                        except Exception as chunk_error:
                            st.error(f"Error embedding chunk: {chunk_error}")
                            st.error(f"Chunk text: {chunk[:100]}...")
                            # Use default similarity score if embedding fails
                            chunk_similarities.append(0.5)
                    
                    # Fill remaining chunks with default similarity
                    while len(chunk_similarities) < len(chunks):
                        chunk_similarities.append(0.5)
                            
                except Exception as embedding_error:
                    st.warning(f"Error creating embeddings: {embedding_error}")
                    # Use default similarity scores if embedding completely fails
                    chunk_similarities = [0.5] * len(chunks)
            else:
                # Use default similarity scores when vector analysis is disabled
                chunk_similarities = [0.5] * len(chunks)
            
            # Create company profile for analysis
            company_profile = {
                "name": "Target Company",
                "industry": company_keywords,
                "products_services": [company_keywords],
                "target_markets": [export_regions]
            }
            
            # Analyze article impact using the hybrid pipeline
            st.write(f"Analyzing article: {article['metadata']['title'][:50]}...")
            
            try:
                impact_analysis = system["pipeline"].analyze_article_impact(
                    article, 
                    company_profile
                )
                
                if "error" in impact_analysis:
                    st.warning(f"Analysis failed: {impact_analysis['error']}")
                    continue
                
                # Check if ml_features exist
                ml_features = {}
                if "impact_analysis" in impact_analysis:
                    impact_obj = impact_analysis["impact_analysis"]
                    
                    if hasattr(impact_obj, 'ml_features'):
                        ml_features = impact_obj.ml_features
                        
                        if ml_features and len(ml_features) > 0:
                            st.success(f"Generated {len(ml_features)} ML features")
                        else:
                            st.warning("ML features dictionary is empty")
                    else:
                        st.warning("impact_analysis object has no ml_features attribute")
                else:
                    st.warning("No impact_analysis in result")
                
                processed_articles.append({
                    "article": article,
                    "impact_analysis": impact_analysis,
                    "ml_features": ml_features
                })
                ml_features_list.append(ml_features)
                    
            except Exception as analysis_error:
                st.error(f"Analysis exception: {str(analysis_error)}")
                continue
            
        except Exception as e:
            st.warning(f"Error processing article: {e}")
            continue
    
    # Calculate average ML features
    if ml_features_list:
        avg_features = {}
        for feature_name in ml_features_list[0].keys():
            avg_value = sum(f[feature_name] for f in ml_features_list) / len(ml_features_list)
            avg_features[feature_name] = avg_value
    else:
        avg_features = {}
        st.warning("No ML features were generated. This could be due to:")
        st.write("- OpenRouter LLM analysis failed")
        st.write("- No articles were successfully analyzed")
        st.write("- API connection issues")
        st.write("Please check the console output for detailed error messages.")
    
    return avg_features, processed_articles, len(ml_features_list)

def main():
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Load system
        with st.spinner("Loading system..."):
            system = load_system()
        
        if system is None:
            st.error("Failed to load system. Please check your .env file.")
            return
        
        # Check API keys
        if not system["settings"].openrouter_api_key:
            st.error("OpenRouter API key is missing. Please set OPENROUTER_API_KEY in your .env file.")
            return
            
        if not system["settings"].newsapi_key:
            st.error("NewsAPI key is missing. Please set NEWSAPI_KEY in your .env file.")
            return
        
        st.success("System loaded successfully")
        
        # Test embedding connection
        with st.spinner("Testing OpenRouter API connection..."):
            try:
                # Simple test with minimal text
                # Test the pipeline
                test_result = system["pipeline"].search_articles("test query", top_k=1)
                st.success("OpenRouter API connection successful")
            except Exception as e:
                st.error(f"OpenRouter API connection failed: {e}")
                st.error("**Possible causes:**")
                st.error("1. Invalid API key")
                st.error("2. API endpoint issue") 
                st.error("3. Rate limiting")
                st.error("4. Network connectivity")
                st.info("You can disable vector similarity analysis in Advanced Options to continue without embeddings.")
                
                # Show API key info (masked)
                api_key = system["settings"].openrouter_api_key
                google_key = system["settings"].google_api_key
                if api_key:
                    st.info(f"API Key format: {api_key[:10]}...{api_key[-4:]}")
                else:
                    st.error("No API key found!")
        
        # Input parameters
        st.header("Input Parameters")
        
        company_keywords = st.text_input(
            "기업 대표 상품/서비스 (쉼표로 구분)",
            placeholder="반도체, 메모리칩, 프로세서",
            help="기업의 주요 제품/서비스를 입력하세요"
        )
        
        export_regions = st.text_input(
            "수출/해외 진출 지역 (쉼표로 구분)",
            placeholder="미국, 중국, 유럽, 일본",
            help="기업이 진출한 지역을 입력하세요"
        )
        
        days_back = st.slider(
            "뉴스 검색 기간 (일)",
            min_value=1,
            max_value=30,
            value=7,
            help="얼마나 과거까지 뉴스를 검색할지 설정하세요 (최대 30일)"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            enable_vector_similarity = st.checkbox(
                "Enable Vector Similarity Analysis",
                value=True,  # 원래대로 True로 복원
                help="Enable vector embedding and similarity analysis (requires OpenRouter API)"
            )
        
        analyze_button = st.button("Feature 추출", type="primary")
    
    # NH농협은행 로고와 헤더 (사이드바 폭 고려한 중앙 정렬)
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; margin: 2rem 0;">
        <div style="margin-bottom: 1rem;">
    """, unsafe_allow_html=True)
    
    try:
        st.image("assets/nh_logo.png", width=300)
    except:
        st.markdown('<div style="color: #1f77b4; font-size: 2.5rem; font-weight: bold;">NH농협은행</div>', unsafe_allow_html=True)
    
    st.markdown("""
        </div>
        <div style="font-size: 2.2rem; font-weight: 600; color: #000000; margin-bottom: 1rem;">NH 기업대출 심사 2.0</div>
        <div style="color: #7f8c8d; font-size: 0.9rem; margin-bottom: 2rem;">프로토타입 v1</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    if analyze_button:
        if not company_keywords or not export_regions:
            st.error("기업 상품/서비스와 수출 지역을 모두 입력해주세요.")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch news
        status_text.text("Fetching news articles...")
        progress_bar.progress(25)
        
        avg_features, processed_articles, num_ml_features = fetch_and_analyze_news(
            system, company_keywords, export_regions, days_back, enable_vector_similarity
        )
        
        if not processed_articles:
            st.error("No relevant articles found. Please try different keywords or regions.")
            return
        
        # Step 2: Display results
        status_text.text("Analyzing results...")
        progress_bar.progress(75)
        
        # Results header
        st.header("Analysis Results")
        
        # ML Features
        st.subheader("ML Features Generated")
        
        if num_ml_features > 0:
            st.info(f"Generated ML features from {num_ml_features} articles")
            
            # Display ML Features table
            if avg_features:
                st.subheader("ML Features Summary")
                features_df = {
                    "Feature": list(avg_features.keys()),
                    "Value": [f"{v:.3f}" for v in avg_features.values()],
                    "Description": [
                        "Market opportunity score (0-1)",
                        "Regulatory risk score (0-1, lower better)",
                        "Competitive pressure score (0-1, lower better)",
                        "Export market health (0-1)",
                        "Policy support score (0-1)",
                        "Overall business outlook (0-1)",
                        "Revenue impact potential (0-1)",
                        "Cost impact potential (0-1, lower better)",
                        "Market share impact (0-1)",
                        "Innovation opportunity (0-1)"
                    ]
                }
                st.dataframe(features_df, use_container_width=True)
        
        # Article Content section
        st.subheader("Article Content")
        
        # Get the first article original content from the analysis results
        article_content = None
        article_title = None
        article_source = None
        article_date = None
        
        if processed_articles and len(processed_articles) > 0:
            first_analysis = processed_articles[0]
            if 'article' in first_analysis:
                article_data = first_analysis['article']
                if 'article' in article_data:
                    article_content = article_data['article'].text
                    article_metadata = article_data['metadata']
                    article_title = article_metadata.get('title', 'N/A')
                    article_source = article_metadata.get('source', 'N/A')
                    article_date = article_metadata.get('published_at', 'N/A')
        
        if article_content:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                <h4 style="color: #1f77b4; margin-top: 0;">{article_title}</h4>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 15px;">
                    <strong>출처:</strong> {article_source} | <strong>발행일:</strong> {article_date}
                </p>
                <div style="background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd;">
                    <p style="margin-bottom: 0; line-height: 1.6; white-space: pre-wrap;">{article_content}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Unable to load article content.")
        
        # Articles analysis
        st.subheader("Source Articles Analysis")
        
        for i, processed in enumerate(processed_articles[:3]):  # Show top 3 articles
            article_data = processed["article"]
            impact_analysis = processed["impact_analysis"]
            ml_features = processed["ml_features"]
            
            # Get article metadata
            metadata = article_data["metadata"]
            similarity_score = article_data["score"]
            
            with st.expander(f"Article {i+1}: {metadata['title'][:60]}... (Similarity: {similarity_score:.3f})", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Source:** {metadata['source']}")
                    st.write(f"**Published:** {metadata['published_at']}")
                    st.write(f"**Similarity Score:** {similarity_score:.3f}")
                    st.write(f"**URL:** {metadata['url']}")
                    
                
                with col2:
                    st.write("**ML Features (0-1 scale):**")
                    for feature, value in ml_features.items():
                        st.write(f"{feature.replace('_', ' ').title()}: {value:.3f}")
        
        
        progress_bar.progress(100)
        status_text.text("Analysis completed!")
        
        # Summary
        st.subheader("Analysis Summary")
        st.write(f"• **Articles Found:** {len(processed_articles)}")
        st.write(f"• **Search Period:** {days_back} days")
        st.write(f"• **Company Keywords:** {company_keywords}")
        st.write(f"• **Export Regions:** {export_regions}")
        st.write(f"• **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
