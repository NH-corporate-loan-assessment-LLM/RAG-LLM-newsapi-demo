#!/usr/bin/env python3
"""
Corporate Loan Assessment RAG System - Streamlit Frontend
회사 키워드 + 해외진출 지역 → ML Feature 분석
"""

import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_settings, validate_settings
from src.company.profile import CompanyProfile
from src.news.newsapi_client import NewsApiIngestor
from src.analysis.openrouter_impact_scorer import OpenRouterImpactScorer
from src.embeddings.openrouter_embedder import OpenRouterEmbedder
from src.vectorstore.faiss_store import FaissStore
from src.chunking.text import chunk_text

# Page config
st.set_page_config(
    page_title="NH농협은행 기업대출심사 RAG 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .article-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load and cache system components"""
    try:
        settings = load_settings()
        validate_settings(settings)
        
        # Initialize components
        news_client = NewsApiIngestor(
            api_key=settings.newsapi_key,
            sources=settings.news_sources,
            domains=settings.news_domains
        )
        
        embedder = OpenRouterEmbedder(api_key=settings.openrouter_api_key)
        vector_store = FaissStore(index_dir=settings.index_dir)
        impact_scorer = OpenRouterImpactScorer(
            openrouter_api_key=settings.openrouter_api_key,
            model="gpt-4o-mini"
        )
        
        return {
            "settings": settings,
            "news_client": news_client,
            "embedder": embedder,
            "vector_store": vector_store,
            "impact_scorer": impact_scorer
        }
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        return None

def generate_search_queries(company_keywords: str, export_regions: str) -> List[str]:
    """Generate search queries based on company keywords and export regions"""
    keywords = [k.strip() for k in company_keywords.split(',') if k.strip()]
    regions = [r.strip() for r in export_regions.split(',') if r.strip()]
    
    queries = []
    
    # 1. Korea-specific policy and trade relations (HIGH PRIORITY)
    for region in regions:
        for keyword in keywords[:2]:  # Top 2 keywords
            # Trade policy and tariffs
            queries.append(f'"{keyword}" "Korea" "{region}" (tariff OR trade agreement OR import ban OR export restriction OR trade war)')
            queries.append(f'"{keyword}" "Korean" "{region}" (policy change OR government decision OR trade dispute OR trade negotiation)')
            queries.append(f'"{keyword}" "South Korea" "{region}" (subsidy OR government support OR market access OR trade barrier)')
            queries.append(f'"{keyword}" "Korea" "{region}" (duty OR customs OR trade barrier OR market opening OR trade liberalization)')
            
            # Market demand and consumer behavior
            queries.append(f'"{keyword}" "Korean" "{region}" (popularity OR demand OR consumer preference OR brand recognition OR market share)')
            queries.append(f'"{keyword}" "Korea" "{region}" (sales OR revenue OR profit OR growth OR market expansion)')
            
            # Regulatory and compliance
            queries.append(f'"{keyword}" "Korean" "{region}" (regulation OR compliance OR standard OR certification OR approval)')
            queries.append(f'"{keyword}" "Korea" "{region}" (safety OR quality OR environmental OR sustainability)')
    
    # 2. Korean product popularity and market trends
    for keyword in keywords[:2]:
        # Global market trends
        queries.append(f'"{keyword}" "Korean" (popular OR demand OR market share OR growth OR trend)')
        queries.append(f'"{keyword}" "Korea" (export OR trade OR business OR revenue OR performance)')
        queries.append(f'"{keyword}" "South Korea" (competition OR new entrant OR market leader OR innovation)')
        
        # Industry-specific trends
        queries.append(f'"{keyword}" "Korean" (technology OR innovation OR R&D OR development OR breakthrough)')
        queries.append(f'"{keyword}" "Korea" (manufacturing OR production OR supply chain OR logistics)')
        queries.append(f'"{keyword}" "South Korea" (investment OR funding OR partnership OR collaboration)')
    
    # 3. Regional market impact and business environment
    for region in regions:
        for keyword in keywords[:1]:  # Top 1 keyword
            # Business impact
            queries.append(f'"{keyword}" "Korea" "{region}" (business impact OR revenue OR profit OR growth OR expansion)')
            queries.append(f'"{keyword}" "Korean" "{region}" (market trend OR consumer preference OR brand recognition OR loyalty)')
            
            # Economic indicators
            queries.append(f'"{keyword}" "Korea" "{region}" (GDP OR economic growth OR inflation OR currency OR exchange rate)')
            queries.append(f'"{keyword}" "Korean" "{region}" (employment OR job creation OR investment OR FDI)')
            
            # Industry analysis
            queries.append(f'"{keyword}" "Korea" "{region}" (industry analysis OR market research OR forecast OR outlook)')
            queries.append(f'"{keyword}" "Korean" "{region}" (competitive analysis OR market positioning OR strategy)')
    
    # 4. Additional credibility-focused queries
    for keyword in keywords[:1]:
        # Government and official sources
        queries.append(f'"{keyword}" "Korea" (government OR ministry OR official OR policy OR regulation)')
        queries.append(f'"{keyword}" "Korean" (trade ministry OR commerce OR industry OR economy)')
        
        # Financial and business news
        queries.append(f'"{keyword}" "Korea" (financial OR business OR corporate OR earnings OR quarterly)')
        queries.append(f'"{keyword}" "Korean" (stock OR market OR investment OR analyst OR rating)')
        
        # International relations
        queries.append(f'"{keyword}" "Korea" (international OR global OR world OR foreign OR diplomatic)')
        queries.append(f'"{keyword}" "Korean" (bilateral OR multilateral OR agreement OR treaty OR partnership)')
    
    return queries

def fetch_and_analyze_news(system, company_keywords: str, export_regions: str, days_back: int = 30):
    """Fetch news and perform RAG analysis"""
    
    # Generate search queries
    queries = generate_search_queries(company_keywords, export_regions)
    
    # Fetch articles
    all_articles = []
    for query in queries:
        try:
            articles = system["news_client"].fetch_articles(
                query=query,
                days_back=days_back,
                language='en',
                page_size=10
            )
            all_articles.extend(articles)
        except Exception as e:
            st.warning(f"Error searching '{query}': {e}")
    
    # Remove duplicates and filter by relevance
    unique_articles = {article.url: article for article in all_articles}.values()
    articles_list = list(unique_articles)
    
    # Filter out articles that are too short or irrelevant
    articles_list = [article for article in articles_list 
                    if len(article.title) > 20 and len(article.content) > 100]
    
    # Sort by relevance (title + content length)
    articles_list.sort(key=lambda x: len(x.title) + len(x.content), reverse=True)
    
    # Limit to top 50 most relevant articles
    articles_list = articles_list[:50]
    
    if not articles_list:
        return None, []
    
    # Process articles with RAG
    processed_articles = []
    ml_features_list = []
    
    for article in articles_list[:10]:  # Top 10 articles
        try:
            # Chunk the article
            article_text = article.to_text()
            chunks = chunk_text(article_text, 1000, 200)
            
            # Create company profile for analysis
            company_profile = {
                "name": "Target Company",
                "industry": company_keywords,
                "products_services": [company_keywords],
                "target_markets": [export_regions]
            }
            
            # Generate ML features
            ml_features = system["impact_scorer"].generate_ml_features(
                article_text, 
                company_profile
            )
            
            processed_articles.append({
                "article": article,
                "chunks": chunks,
                "ml_features": ml_features
            })
            ml_features_list.append(ml_features)
            
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
    
    return avg_features, processed_articles

def main():
    # Header
    st.markdown('<div class="main-header">🏦 Corporate Loan Assessment RAG System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load system
        with st.spinner("Loading system..."):
            system = load_system()
        
        if system is None:
            st.error("Failed to load system. Please check your .env file.")
            return
        
        st.success("✅ System loaded successfully")
        
        # Input parameters
        st.header("📝 Input Parameters")
        
        company_keywords = st.text_input(
            "Company Keywords (comma-separated)",
            placeholder="semiconductor, memory chips, processors",
            help="Enter the main products/services of the company"
        )
        
        export_regions = st.text_input(
            "Export/International Regions (comma-separated)",
            placeholder="US, China, EU, Japan",
            help="Enter the regions where the company operates"
        )
        
        days_back = st.slider(
            "News Search Period (days)",
            min_value=7,
            max_value=90,
            value=30,
            help="How far back to search for news"
        )
        
        analyze_button = st.button("🔍 Analyze Company", type="primary")
    
    # Main content
    if analyze_button:
        if not company_keywords or not export_regions:
            st.error("Please enter both company keywords and export regions.")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch news
        status_text.text("📰 Fetching news articles...")
        progress_bar.progress(25)
        
        avg_features, processed_articles = fetch_and_analyze_news(
            system, company_keywords, export_regions, days_back
        )
        
        if not processed_articles:
            st.error("No relevant articles found. Please try different keywords or regions.")
            return
        
        # Step 2: Display results
        status_text.text("📊 Analyzing results...")
        progress_bar.progress(75)
        
        # Results header
        st.header("📈 Analysis Results")
        
        # ML Features
        st.subheader("🤖 ML Features for Loan Assessment")
        
        if avg_features:
            # Create columns for features
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Market Opportunity",
                    f"{avg_features.get('market_opportunity_score', 0):.3f}",
                    help="0-1 scale, higher is better"
                )
                st.metric(
                    "Export Market Health",
                    f"{avg_features.get('export_market_health', 0):.3f}",
                    help="0-1 scale, higher is better"
                )
                st.metric(
                    "Policy Support",
                    f"{avg_features.get('policy_support_score', 0):.3f}",
                    help="0-1 scale, higher is better"
                )
            
            with col2:
                st.metric(
                    "Regulatory Risk",
                    f"{avg_features.get('regulatory_risk_score', 0):.3f}",
                    help="0-1 scale, lower is better"
                )
                st.metric(
                    "Competitive Pressure",
                    f"{avg_features.get('competitive_pressure_score', 0):.3f}",
                    help="0-1 scale, lower is better"
                )
                st.metric(
                    "Overall Business Outlook",
                    f"{avg_features.get('overall_business_outlook', 0):.3f}",
                    help="0-1 scale, higher is better"
                )
            
            with col3:
                st.metric(
                    "Revenue Impact Potential",
                    f"{avg_features.get('revenue_impact_potential', 0):.3f}",
                    help="0-1 scale, higher is better"
                )
                st.metric(
                    "Cost Impact Potential",
                    f"{avg_features.get('cost_impact_potential', 0):.3f}",
                    help="0-1 scale, lower is better"
                )
                st.metric(
                    "Innovation Opportunity",
                    f"{avg_features.get('innovation_opportunity', 0):.3f}",
                    help="0-1 scale, higher is better"
                )
            
            # Detailed features table
            st.subheader("📋 Detailed ML Features")
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
        
        # Articles analysis
        st.subheader("📰 Source Articles Analysis")
        
        for i, processed in enumerate(processed_articles[:5]):  # Show top 5 articles
            article = processed["article"]
            ml_features = processed["ml_features"]
            
            with st.expander(f"📄 Article {i+1}: {article.title[:80]}..."):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Source:** {article.source}")
                    st.write(f"**Published:** {article.published_at}")
                    st.write(f"**URL:** {article.url}")
                    st.write(f"**Content:** {article.to_text()[:500]}...")
                
                with col2:
                    st.write("**ML Features from this article:**")
                    for feature, value in ml_features.items():
                        st.write(f"• {feature.replace('_', ' ').title()}: {value:.3f}")
        
        # Loan recommendation
        st.subheader("🏦 Loan Assessment Recommendation")
        
        if avg_features:
            overall_score = avg_features.get('overall_business_outlook', 0.5)
            
            if overall_score > 0.6:
                assessment = "🟢 POSITIVE"
                recommendation = "Consider loan approval with standard terms"
                color = "green"
            elif overall_score < 0.4:
                assessment = "🔴 NEGATIVE"
                recommendation = "Require additional due diligence and higher interest rates"
                color = "red"
            else:
                assessment = "🟡 NEUTRAL"
                recommendation = "Standard review process with monitoring"
                color = "orange"
            
            st.markdown(f"""
            <div class="metric-container">
                <h3>Overall Assessment: <span style="color: {color}">{assessment}</span></h3>
                <p><strong>Score:</strong> {overall_score:.3f}</p>
                <p><strong>Recommendation:</strong> {recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis completed!")
        
        # Summary
        st.subheader("📊 Analysis Summary")
        st.write(f"• **Articles Found:** {len(processed_articles)}")
        st.write(f"• **Search Period:** {days_back} days")
        st.write(f"• **Company Keywords:** {company_keywords}")
        st.write(f"• **Export Regions:** {export_regions}")
        st.write(f"• **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
