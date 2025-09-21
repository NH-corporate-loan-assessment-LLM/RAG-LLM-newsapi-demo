#!/usr/bin/env python3
"""
Corporate Loan Assessment RAG System
Main application for analyzing news impact on corporate loan decisions
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

from src.config import load_settings, validate_settings
from src.company import CompanyProfile, create_sample_company
from src.news import NewsApiIngestor
from src.rag import CorporateRagPipeline


def setup_environment():
    """Setup environment and validate configuration"""
    print("🔧 Setting up Corporate Loan Assessment RAG System...")
    
    # Load settings
    settings = load_settings()
    
    try:
        validate_settings(settings)
        print("✅ Configuration validated successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\nRequired environment variables:")
        print("- GOOGLE_API_KEY: Your Google Gemini API key")
        print("- NEWSAPI_KEY: Your NewsAPI key")
        print("\nOptional environment variables:")
        print("- INDEX_DIR: Directory for vector store (default: .faiss_index)")
        print("- NEWS_SOURCES: Comma-separated news sources")
        print("- NEWS_DOMAINS: Comma-separated news domains")
        sys.exit(1)
    
    return settings


def collect_news_data(settings, company_profile: CompanyProfile) -> int:
    """Collect and index news data for the company"""
    print(f"\n📰 Collecting news data for {company_profile.company_name}...")
    
    # Initialize news ingestor
    news_ingestor = NewsApiIngestor(
        api_key=settings.newsapi_key,
        sources=settings.news_sources,
        domains=settings.news_domains
    )
    
    # Initialize RAG pipeline
    rag_pipeline = CorporateRagPipeline(
        openrouter_api_key=settings.openrouter_api_key,
        index_dir=settings.index_dir,
        settings=settings
    )
    
    total_chunks = 0
    
    # Collect different types of news
    news_types = [
        ("Company News", news_ingestor.fetch_company_news),
        ("Policy News", news_ingestor.fetch_policy_news),
        ("Market Trends", news_ingestor.fetch_market_trends)
    ]
    
    for news_type, fetch_func in news_types:
        print(f"  📊 Fetching {news_type.lower()}...")
        
        try:
            articles = fetch_func(company_profile, days_back=90, max_articles=100)
            print(f"    Found {len(articles)} articles")
            
            if articles:
                # Convert to format for indexing
                article_data = []
                for article in articles:
                    article_data.append((
                        article.to_text(),
                        article.to_metadata()
                    ))
                
                # Index articles
                chunks_added = rag_pipeline.upsert_news_articles(article_data)
                total_chunks += chunks_added
                print(f"    Indexed {chunks_added} chunks")
            
        except Exception as e:
            print(f"    ❌ Error fetching {news_type.lower()}: {e}")
    
    print(f"✅ Total chunks indexed: {total_chunks}")
    return total_chunks


def analyze_company(settings, company_profile: CompanyProfile) -> Dict[str, Any]:
    """Analyze company for loan assessment"""
    print(f"\n🔍 Analyzing {company_profile.company_name} for loan assessment...")
    
    # Initialize RAG pipeline
    rag_pipeline = CorporateRagPipeline(
        openrouter_api_key=settings.openrouter_api_key,
        index_dir=settings.index_dir,
        settings=settings
    )
    
    # Perform analysis
    analysis_result = rag_pipeline.analyze_company_impact(company_profile)
    
    if analysis_result["status"] == "success":
        print("✅ Analysis completed successfully")
        
        # Display summary
        summary = analysis_result["impact_summary"]
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"  Overall Assessment: {summary['overall_assessment']}")
        print(f"  Articles Analyzed: {summary['total_articles_analyzed']}")
        print(f"  Average Impact: {summary['average_scores']['overall_impact']}/100")
        print(f"  Average Sentiment: {summary['average_scores']['sentiment']}/100")
        print(f"  Recommendation: {summary['loan_assessment_recommendation']}")
        
        # Display ML features
        ml_features = analysis_result["ml_features"]
        print(f"\n🤖 ML FEATURES:")
        for key, value in ml_features.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    else:
        print(f"❌ Analysis failed: {analysis_result['message']}")
    
    return analysis_result


def generate_report(settings, company_profile: CompanyProfile) -> str:
    """Generate comprehensive loan assessment report"""
    print(f"\n📋 Generating loan assessment report for {company_profile.company_name}...")
    
    # Initialize RAG pipeline
    rag_pipeline = CorporateRagPipeline(
        openrouter_api_key=settings.openrouter_api_key,
        index_dir=settings.index_dir,
        settings=settings
    )
    
    # Generate report
    report = rag_pipeline.generate_loan_assessment_report(company_profile)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"loan_assessment_{company_profile.company_name.replace(' ', '_')}_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {filename}")
    return report


def interactive_mode(settings):
    """Interactive mode for testing the system"""
    print("\n🎯 Interactive Mode - Corporate Loan Assessment")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Use sample company (Samsung Electronics)")
        print("2. Create custom company profile")
        print("3. Collect news data")
        print("4. Analyze company")
        print("5. Generate report")
        print("6. View vector store stats")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            company = create_sample_company()
            print(f"✅ Using sample company: {company.company_name}")
            
        elif choice == "2":
            company = create_custom_company()
            
        elif choice == "3":
            if 'company' not in locals():
                print("❌ Please select a company first (option 1 or 2)")
                continue
            collect_news_data(settings, company)
            
        elif choice == "4":
            if 'company' not in locals():
                print("❌ Please select a company first (option 1 or 2)")
                continue
            analyze_company(settings, company)
            
        elif choice == "5":
            if 'company' not in locals():
                print("❌ Please select a company first (option 1 or 2)")
                continue
            generate_report(settings, company)
            
        elif choice == "6":
            from src.vectorstore import FaissStore
            store = FaissStore(settings.index_dir)
            stats = store.get_stats()
            print(f"\n📊 Vector Store Statistics:")
            print(json.dumps(stats, indent=2))
            
        elif choice == "7":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")


def create_custom_company() -> CompanyProfile:
    """Create a custom company profile interactively"""
    print("\n🏢 Create Custom Company Profile")
    print("-" * 30)
    
    company_name = input("Company Name: ").strip()
    industry = input("Industry: ").strip()
    
    print("Main Products/Services (comma-separated):")
    products = [p.strip() for p in input().split(",") if p.strip()]
    
    print("Target Markets (comma-separated, e.g., US,China,EU):")
    markets = [m.strip() for m in input().split(",") if m.strip()]
    
    business_model = input("Business Model (optional): ").strip() or None
    
    try:
        annual_revenue = float(input("Annual Revenue in USD (optional): ").strip() or "0")
        if annual_revenue == 0:
            annual_revenue = None
    except ValueError:
        annual_revenue = None
    
    try:
        employee_count = int(input("Employee Count (optional): ").strip() or "0")
        if employee_count == 0:
            employee_count = None
    except ValueError:
        employee_count = None
    
    return CompanyProfile(
        company_name=company_name,
        industry=industry,
        main_products_services=products,
        target_markets=markets,
        business_model=business_model,
        annual_revenue=annual_revenue,
        employee_count=employee_count
    )


def main():
    """Main application entry point"""
    print("🏦 Corporate Loan Assessment RAG System")
    print("=" * 50)
    
    # Setup environment
    settings = setup_environment()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            # Demo mode with sample company
            print("\n🚀 Running in demo mode...")
            company = create_sample_company()
            
            print(f"📊 Demo Company: {company.company_name}")
            print(f"   Industry: {company.industry}")
            print(f"   Products: {', '.join(company.main_products_services)}")
            print(f"   Markets: {', '.join(company.target_markets)}")
            
            # Collect news data
            collect_news_data(settings, company)
            
            # Analyze company
            analysis_result = analyze_company(settings, company)
            
            # Generate report
            generate_report(settings, company)
            
            print("\n✅ Demo completed successfully!")
            
        elif sys.argv[1] == "--interactive":
            # Interactive mode
            interactive_mode(settings)
            
        else:
            print("Usage: python main.py [--demo|--interactive]")
            sys.exit(1)
    else:
        # Default to interactive mode
        interactive_mode(settings)


if __name__ == "__main__":
    main()