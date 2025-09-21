from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import OpenAI


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


class OpenRouterImpactScorer:
    """Analyzes news articles for their impact on company loan assessment using OpenRouter"""
    
    def __init__(self, openrouter_api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
    
    def generate_ml_features(self, article_text: str, company_profile: Dict[str, Any]) -> Dict[str, float]:
        """
        LLM이 기사를 분석하여 ML 모델용 feature 생성
        
        Args:
            article_text: 뉴스 기사 텍스트
            company_profile: 기업 프로필 정보
            
        Returns:
            ML 모델용 feature 딕셔너리 (0-1 스케일)
        """
        prompt = f"""
        Analyze this news article for {company_profile.get('name', 'the company')} and generate ML features for loan assessment.
        
        Company Information:
        - Name: {company_profile.get('name', 'Unknown')}
        - Industry: {company_profile.get('industry', 'Unknown')}
        - Main Products: {', '.join(company_profile.get('products_services', []))}
        - Export Markets: {', '.join(company_profile.get('target_markets', []))}
        
        News Article:
        {article_text}
        
        Generate the following ML features (0-1 scale, where 1 is most positive):
        
        1. market_opportunity_score: 시장 기회 점수 (0-1)
        2. regulatory_risk_score: 규제 리스크 점수 (0-1, 높을수록 위험)
        3. competitive_pressure_score: 경쟁 압박 점수 (0-1, 높을수록 압박)
        4. export_market_health: 수출시장 건강도 (0-1)
        5. policy_support_score: 정책 지원 점수 (0-1)
        6. overall_business_outlook: 전체 사업 전망 (0-1)
        7. revenue_impact_potential: 매출 영향 잠재력 (0-1)
        8. cost_impact_potential: 비용 영향 잠재력 (0-1, 높을수록 비용 증가)
        9. market_share_impact: 시장점유율 영향 (0-1)
        10. innovation_opportunity: 혁신 기회 (0-1)
        
        Consider:
        - Direct business impact (revenue, costs, market share)
        - Regulatory/policy changes affecting the company
        - Market trends and competitive dynamics
        - Industry-specific factors and cycles
        - Long-term vs short-term implications
        
        Return only a JSON object with the feature values:
        {{
            "market_opportunity_score": 0.0,
            "regulatory_risk_score": 0.0,
            "competitive_pressure_score": 0.0,
            "export_market_health": 0.0,
            "policy_support_score": 0.0,
            "overall_business_outlook": 0.0,
            "revenue_impact_potential": 0.0,
            "cost_impact_potential": 0.0,
            "market_share_impact": 0.0,
            "innovation_opportunity": 0.0
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in corporate loan assessment. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            features = json.loads(content)
            return features
        except Exception as e:
            print(f"Error generating ML features: {e}")
            # Return default neutral features
            return {
                "market_opportunity_score": 0.5,
                "regulatory_risk_score": 0.5,
                "competitive_pressure_score": 0.5,
                "export_market_health": 0.5,
                "policy_support_score": 0.5,
                "overall_business_outlook": 0.5,
                "revenue_impact_potential": 0.5,
                "cost_impact_potential": 0.5,
                "market_share_impact": 0.5,
                "innovation_opportunity": 0.5
            }
    
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
                    {"role": "system", "content": "You are a financial analyst specializing in corporate loan assessment. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            analysis_text = response.choices[0].message.content or ""
            
            # Parse the structured response
            impact_score = self._parse_impact_analysis(analysis_text, article_metadata)
            
            # Generate ML features
            impact_score.ml_features = self.generate_ml_features(article_text, company_profile.__dict__)
            
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

        OUTPUT FORMAT (JSON):
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
            company_name=company_profile.name,
            industry=company_profile.industry,
            products=", ".join(company_profile.products_services),
            markets=", ".join(company_profile.target_markets),
            business_model=getattr(company_profile, 'business_model', 'Not specified'),
            title=article_metadata.get("title", ""),
            source=article_metadata.get("source_name", ""),
            published_date=article_metadata.get("published_at", ""),
            content=article_text[:2000]  # Limit content length
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


# Alias for backward compatibility
ImpactScorer = OpenRouterImpactScorer
