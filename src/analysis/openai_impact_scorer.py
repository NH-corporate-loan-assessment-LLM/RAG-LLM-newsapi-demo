from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

import openai
from langchain_core.prompts import PromptTemplate


@dataclass
class ImpactScore:
    """Impact score for a news article on company assessment"""
    
    article_title: str
    article_source: str
    published_date: str
    
    # Impact scores (0-100)
    overall_impact: float
    market_impact: float
    policy_impact: float
    competitive_impact: float
    
    # Sentiment scores (-100 to +100)
    sentiment_score: float
    confidence_score: float
    
    # Categorization
    impact_type: str  # "positive", "negative", "neutral"
    relevance_score: float  # How relevant to the company
    
    # Analysis details
    key_factors: List[str]
    risk_factors: List[str]
    opportunity_factors: List[str]
    
    # ML Features for loan assessment
    ml_features: Dict[str, float]


class OpenAIImpactScorer:
    """OpenAI-based impact analysis for news articles on company loan assessment"""
    
    def __init__(self, api_key: str, positive_keywords: List[str], 
                 negative_keywords: List[str], neutral_keywords: List[str]):
        self.client = openai.OpenAI(api_key=api_key)
        self.positive_keywords = positive_keywords
        self.negative_keywords = negative_keywords
        self.neutral_keywords = neutral_keywords
        
        # Use GPT-4 for better analysis
        self.model = "gpt-4o-mini"  # Cost-effective but powerful
    
    def analyze_article_impact(
        self, 
        article_text: str, 
        article_metadata: Dict[str, Any],
        company_profile
    ) -> ImpactScore:
        """Analyze the impact of a single article on company assessment"""
        
        # Create analysis prompt
        prompt = self._create_impact_analysis_prompt(article_text, article_metadata, company_profile)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in corporate loan assessment. Provide detailed, accurate analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content or ""
            
            # Parse the structured response
            impact_score = self._parse_impact_analysis(analysis_text, article_metadata)
            
            # Generate ML features
            impact_score.ml_features = self._generate_ml_features(impact_score, company_profile)
            
            return impact_score
            
        except Exception as e:
            print(f"Error analyzing article impact: {e}")
            return self._create_default_impact_score(article_metadata)
    
    def _create_impact_analysis_prompt(
        self, 
        article_text: str, 
        article_metadata: Dict[str, Any],
        company_profile
    ) -> str:
        """Create prompt for impact analysis"""
        
        template = """
        You are a financial analyst specializing in corporate loan assessment. 
        Analyze the following news article for its potential impact on a company's loan application.

        COMPANY PROFILE:
        - Name: {company_name}
        - Industry: {industry}
        - Main Products/Services: {products}
        - Target Markets: {markets}
        - Business Model: {business_model}

        NEWS ARTICLE:
        Title: {title}
        Source: {source}
        Published: {published_date}
        Content: {content}

        ANALYSIS REQUIREMENTS:
        1. Assess overall impact on company's loanworthiness (0-100 scale)
        2. Evaluate specific impact areas:
           - Market Impact: How does this affect the company's target markets?
           - Policy Impact: How do policy/regulatory changes affect the company?
           - Competitive Impact: How does this affect competitive position?
        3. Determine sentiment (-100 to +100, where +100 is very positive)
        4. Assess confidence in analysis (0-100)
        5. Categorize impact type: positive, negative, or neutral
        6. Calculate relevance to company (0-100)

        OUTPUT FORMAT (JSON only, no other text):
        {{
            "overall_impact": <0-100>,
            "market_impact": <0-100>,
            "policy_impact": <0-100>,
            "competitive_impact": <0-100>,
            "sentiment_score": <-100 to +100>,
            "confidence_score": <0-100>,
            "impact_type": "<positive|negative|neutral>",
            "relevance_score": <0-100>,
            "key_factors": ["factor1", "factor2", ...],
            "risk_factors": ["risk1", "risk2", ...],
            "opportunity_factors": ["opp1", "opp2", ...]
        }}

        Focus on factors that would influence a bank's lending decision.
        """
        
        return template.format(
            company_name=company_profile.company_name,
            industry=company_profile.industry,
            products=", ".join(company_profile.main_products_services),
            markets=", ".join(company_profile.target_markets),
            business_model=company_profile.business_model or "Not specified",
            title=article_metadata.get("title", ""),
            source=article_metadata.get("source_name", ""),
            published_date=article_metadata.get("published_at", ""),
            content=article_text[:3000]  # Limit content length for token efficiency
        )
    
    def _parse_impact_analysis(self, analysis_text: str, article_metadata: Dict[str, Any]) -> ImpactScore:
        """Parse the structured analysis response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            return ImpactScore(
                article_title=article_metadata.get("title", ""),
                article_source=article_metadata.get("source_name", ""),
                published_date=article_metadata.get("published_at", ""),
                overall_impact=float(data.get("overall_impact", 50)),
                market_impact=float(data.get("market_impact", 50)),
                policy_impact=float(data.get("policy_impact", 50)),
                competitive_impact=float(data.get("competitive_impact", 50)),
                sentiment_score=float(data.get("sentiment_score", 0)),
                confidence_score=float(data.get("confidence_score", 50)),
                impact_type=data.get("impact_type", "neutral"),
                relevance_score=float(data.get("relevance_score", 50)),
                key_factors=data.get("key_factors", []),
                risk_factors=data.get("risk_factors", []),
                opportunity_factors=data.get("opportunity_factors", []),
                ml_features={}  # Will be filled later
            )
            
        except Exception as e:
            print(f"Error parsing impact analysis: {e}")
            return self._create_default_impact_score(article_metadata)
    
    def _generate_ml_features(self, impact_score: ImpactScore, company_profile) -> Dict[str, float]:
        """Generate ML features for loan assessment model"""
        
        features = {
            # Impact scores (normalized 0-1)
            "overall_impact_norm": impact_score.overall_impact / 100.0,
            "market_impact_norm": impact_score.market_impact / 100.0,
            "policy_impact_norm": impact_score.policy_impact / 100.0,
            "competitive_impact_norm": impact_score.competitive_impact / 100.0,
            
            # Sentiment (normalized -1 to 1)
            "sentiment_norm": impact_score.sentiment_score / 100.0,
            
            # Confidence and relevance
            "confidence_norm": impact_score.confidence_score / 100.0,
            "relevance_norm": impact_score.relevance_score / 100.0,
            
            # Impact type encoding
            "is_positive_impact": 1.0 if impact_score.impact_type == "positive" else 0.0,
            "is_negative_impact": 1.0 if impact_score.impact_type == "negative" else 0.0,
            "is_neutral_impact": 1.0 if impact_score.impact_type == "neutral" else 0.0,
            
            # Factor counts
            "num_key_factors": len(impact_score.key_factors) / 10.0,  # Normalize
            "num_risk_factors": len(impact_score.risk_factors) / 10.0,
            "num_opportunity_factors": len(impact_score.opportunity_factors) / 10.0,
            
            # Risk/Opportunity ratio
            "risk_opportunity_ratio": (
                len(impact_score.risk_factors) / 
                max(1, len(impact_score.opportunity_factors))
            ),
        }
        
        # Add company-specific features
        if company_profile.annual_revenue:
            features["revenue_scale"] = min(company_profile.annual_revenue / 1e9, 10.0)  # Cap at 10B
        
        if company_profile.market_share:
            features["market_share_norm"] = company_profile.market_share / 100.0
        
        return features
    
    def _create_default_impact_score(self, article_metadata: Dict[str, Any]) -> ImpactScore:
        """Create default impact score when analysis fails"""
        return ImpactScore(
            article_title=article_metadata.get("title", ""),
            article_source=article_metadata.get("source_name", ""),
            published_date=article_metadata.get("published_at", ""),
            overall_impact=50.0,
            market_impact=50.0,
            policy_impact=50.0,
            competitive_impact=50.0,
            sentiment_score=0.0,
            confidence_score=25.0,  # Low confidence for default
            impact_type="neutral",
            relevance_score=25.0,  # Low relevance for default
            key_factors=[],
            risk_factors=[],
            opportunity_factors=[],
            ml_features={}
        )
    
    def aggregate_impact_scores(self, impact_scores: List[ImpactScore]) -> Dict[str, float]:
        """Aggregate multiple impact scores into overall assessment"""
        if not impact_scores:
            return {}
        
        # Weight by confidence and relevance
        total_weight = 0
        weighted_scores = {
            "overall_impact": 0,
            "market_impact": 0,
            "policy_impact": 0,
            "competitive_impact": 0,
            "sentiment_score": 0,
        }
        
        for score in impact_scores:
            weight = (score.confidence_score * score.relevance_score) / 10000.0
            total_weight += weight
            
            weighted_scores["overall_impact"] += score.overall_impact * weight
            weighted_scores["market_impact"] += score.market_impact * weight
            weighted_scores["policy_impact"] += score.policy_impact * weight
            weighted_scores["competitive_impact"] += score.competitive_impact * weight
            weighted_scores["sentiment_score"] += score.sentiment_score * weight
        
        if total_weight > 0:
            for key in weighted_scores:
                weighted_scores[key] /= total_weight
        
        # Add aggregate ML features
        weighted_scores.update({
            "num_articles_analyzed": len(impact_scores),
            "avg_confidence": sum(s.confidence_score for s in impact_scores) / len(impact_scores),
            "avg_relevance": sum(s.relevance_score for s in impact_scores) / len(impact_scores),
            "positive_impact_ratio": sum(1 for s in impact_scores if s.impact_type == "positive") / len(impact_scores),
            "negative_impact_ratio": sum(1 for s in impact_scores if s.impact_type == "negative") / len(impact_scores),
        })
        
        return weighted_scores

