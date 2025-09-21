from __future__ import annotations

from typing import List, Tuple, Dict, Any
import re
from datetime import datetime, timezone

from openai import OpenAI

from src.chunking.text import chunk_text
from src.embeddings import OpenRouterEmbedder
from src.vectorstore import FaissStore
from src.analysis import ImpactScorer


class CorporateRagPipeline:
    """RAG pipeline for corporate loan assessment using news analysis"""
    
    def __init__(self, openrouter_api_key: str, index_dir: str, settings, use_local_perplexity: bool = True):
        self.embedder = OpenRouterEmbedder(api_key=openrouter_api_key)
        self.store = FaissStore(index_dir=index_dir)
        self.impact_scorer = ImpactScorer(
            openrouter_api_key=openrouter_api_key,
            model="gpt-4o-mini"
        )
        
        # Use simple text chunking (Discord Chatbot style)
        print("Using simple text chunking")
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        self.settings = settings

    def upsert_news_articles(self, articles: List[Tuple[str, Dict[str, Any]]]) -> int:
        """Add news articles to the vector store"""
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        
        for article_text, metadata in articles:
            # Use simple text chunking (Discord Chatbot style)
            chunks = chunk_text(article_text, self.chunk_size, self.chunk_overlap)
            
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append(metadata)
        
        if not texts:
            return 0
        
        # Generate embeddings
        vectors = self.embedder.embed_texts(texts)
        
        # Add to vector store
        self.store.add_texts(texts, vectors, metadatas)
        
        return len(texts)

    def search_relevant_news(self, query: str, top_k: int = None) -> List[Any]:
        """Search for relevant news articles"""
        if top_k is None:
            top_k = self.settings.top_k_results
        
        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector, top_k=top_k)

    def search_company_news(self, company_profile, top_k: int = None) -> List[Any]:
        """Search for news relevant to a specific company"""
        if top_k is None:
            top_k = self.settings.top_k_results
        
        # Create company-specific search query
        company_context = self.embedder.embed_company_context(company_profile)
        
        # Search with company context
        results = self.store.search(company_context, top_k=top_k)
        
        # Also search with company keywords
        keywords = company_profile.get_search_keywords()
        keyword_results = []
        
        for keyword in keywords[:5]:  # Limit to avoid too many searches
            keyword_vector = self.embedder.embed_query(keyword)
            keyword_results.extend(self.store.search(keyword_vector, top_k=3))
        
        # Combine and deduplicate results
        all_results = results + keyword_results
        seen_texts = set()
        unique_results = []
        
        for result in all_results:
            if result.text not in seen_texts:
                seen_texts.add(result.text)
                unique_results.append(result)
        
        return unique_results[:top_k]

    def analyze_company_impact(self, company_profile) -> Dict[str, Any]:
        """Analyze news impact on company for loan assessment"""
        
        # Search for relevant news
        relevant_news = self.search_company_news(company_profile, top_k=20)
        
        if not relevant_news:
            return {
                "status": "no_relevant_news",
                "message": "No relevant news found for analysis",
                "ml_features": {},
                "impact_summary": {}
            }
        
        # Analyze impact of each article
        impact_scores = []
        for news_item in relevant_news:
            try:
                impact_score = self.impact_scorer.analyze_article_impact(
                    news_item.text,
                    news_item.metadata,
                    company_profile
                )
                impact_scores.append(impact_score)
            except Exception as e:
                print(f"Error analyzing impact for article: {e}")
                continue
        
        if not impact_scores:
            return {
                "status": "analysis_failed",
                "message": "Failed to analyze news impact",
                "ml_features": {},
                "impact_summary": {}
            }
        
        # Aggregate impact scores
        aggregated_features = self.impact_scorer.aggregate_impact_scores(impact_scores)
        
        # Generate summary
        impact_summary = self._generate_impact_summary(impact_scores, company_profile)
        
        return {
            "status": "success",
            "message": f"Analyzed {len(impact_scores)} relevant articles",
            "ml_features": aggregated_features,
            "impact_summary": impact_summary,
            "detailed_scores": [
                {
                    "title": score.article_title,
                    "source": score.article_source,
                    "impact_type": score.impact_type,
                    "overall_impact": score.overall_impact,
                    "sentiment": score.sentiment_score,
                    "relevance": score.relevance_score
                }
                for score in impact_scores
            ]
        }

    def _generate_impact_summary(self, impact_scores: List[Any], company_profile) -> Dict[str, Any]:
        """Generate human-readable impact summary"""
        
        # Count impact types
        positive_count = sum(1 for s in impact_scores if s.impact_type == "positive")
        negative_count = sum(1 for s in impact_scores if s.impact_type == "negative")
        neutral_count = sum(1 for s in impact_scores if s.impact_type == "neutral")
        
        # Calculate averages
        avg_overall_impact = sum(s.overall_impact for s in impact_scores) / len(impact_scores)
        avg_sentiment = sum(s.sentiment_score for s in impact_scores) / len(impact_scores)
        avg_relevance = sum(s.relevance_score for s in impact_scores) / len(impact_scores)
        
        # Determine overall assessment
        if avg_sentiment > 20 and positive_count > negative_count:
            overall_assessment = "POSITIVE"
        elif avg_sentiment < -20 and negative_count > positive_count:
            overall_assessment = "NEGATIVE"
        else:
            overall_assessment = "NEUTRAL"
        
        # Collect key insights
        all_key_factors = []
        all_risk_factors = []
        all_opportunity_factors = []
        
        for score in impact_scores:
            all_key_factors.extend(score.key_factors)
            all_risk_factors.extend(score.risk_factors)
            all_opportunity_factors.extend(score.opportunity_factors)
        
        # Get most common factors
        from collections import Counter
        top_key_factors = [item for item, count in Counter(all_key_factors).most_common(5)]
        top_risk_factors = [item for item, count in Counter(all_risk_factors).most_common(3)]
        top_opportunity_factors = [item for item, count in Counter(all_opportunity_factors).most_common(3)]
        
        return {
            "overall_assessment": overall_assessment,
            "total_articles_analyzed": len(impact_scores),
            "impact_breakdown": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            },
            "average_scores": {
                "overall_impact": round(avg_overall_impact, 1),
                "sentiment": round(avg_sentiment, 1),
                "relevance": round(avg_relevance, 1)
            },
            "key_insights": {
                "top_factors": top_key_factors,
                "main_risks": top_risk_factors,
                "main_opportunities": top_opportunity_factors
            },
            "loan_assessment_recommendation": self._get_loan_recommendation(
                overall_assessment, avg_overall_impact, avg_sentiment
            )
        }

    def _get_loan_recommendation(self, assessment: str, impact: float, sentiment: float) -> str:
        """Generate loan assessment recommendation"""
        
        if assessment == "POSITIVE" and impact > 70 and sentiment > 30:
            return "STRONG POSITIVE - Favorable market conditions and positive sentiment support loan approval"
        elif assessment == "NEGATIVE" and impact > 70 and sentiment < -30:
            return "STRONG NEGATIVE - Adverse market conditions and negative sentiment suggest caution"
        elif assessment == "POSITIVE":
            return "POSITIVE - Generally favorable conditions with some positive indicators"
        elif assessment == "NEGATIVE":
            return "NEGATIVE - Some concerning factors that require additional review"
        else:
            return "NEUTRAL - Mixed signals, recommend additional due diligence"

    def generate_loan_assessment_report(self, company_profile) -> str:
        """Generate comprehensive loan assessment report"""
        
        analysis_result = self.analyze_company_impact(company_profile)
        
        if analysis_result["status"] != "success":
            return f"Assessment Status: {analysis_result['status']}\nMessage: {analysis_result['message']}"
        
        impact_summary = analysis_result["impact_summary"]
        ml_features = analysis_result["ml_features"]
        
        # Create detailed report
        report = f"""
# CORPORATE LOAN ASSESSMENT REPORT

## Company: {company_profile.company_name}
## Industry: {company_profile.industry}
## Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
Overall Assessment: {impact_summary['overall_assessment']}
Recommendation: {impact_summary['loan_assessment_recommendation']}

## NEWS ANALYSIS RESULTS
- Total Articles Analyzed: {impact_summary['total_articles_analyzed']}
- Impact Breakdown: {impact_summary['impact_breakdown']}
- Average Impact Score: {impact_summary['average_scores']['overall_impact']}/100
- Average Sentiment: {impact_summary['average_scores']['sentiment']}/100
- Average Relevance: {impact_summary['average_scores']['relevance']}/100

## KEY INSIGHTS
### Top Factors:
{chr(10).join(f"- {factor}" for factor in impact_summary['key_insights']['top_factors'])}

### Main Risks:
{chr(10).join(f"- {risk}" for risk in impact_summary['key_insights']['main_risks'])}

### Main Opportunities:
{chr(10).join(f"- {opp}" for opp in impact_summary['key_insights']['main_opportunities'])}

## ML FEATURES FOR LOAN MODEL
- Overall Impact (normalized): {ml_features.get('overall_impact_norm', 0):.3f}
- Market Impact (normalized): {ml_features.get('market_impact_norm', 0):.3f}
- Policy Impact (normalized): {ml_features.get('policy_impact_norm', 0):.3f}
- Sentiment Score (normalized): {ml_features.get('sentiment_norm', 0):.3f}
- Positive Impact Ratio: {ml_features.get('positive_impact_ratio', 0):.3f}
- Risk/Opportunity Ratio: {ml_features.get('risk_opportunity_ratio', 0):.3f}

## DETAILED ARTICLE ANALYSIS
"""
        
        # Add detailed article analysis
        for i, article in enumerate(analysis_result["detailed_scores"][:10], 1):
            report += f"""
### Article {i}: {article['title'][:80]}...
- Source: {article['source']}
- Impact Type: {article['impact_type'].upper()}
- Overall Impact: {article['overall_impact']}/100
- Sentiment: {article['sentiment']}/100
- Relevance: {article['relevance']}/100
"""
        
        return report
