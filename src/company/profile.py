from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class CompanyProfile:
    """Company profile for loan assessment analysis"""
    
    # Basic Information
    company_name: str
    industry: str
    main_products_services: List[str]
    target_markets: List[str]  # Geographic markets
    
    # Business Details
    business_model: Optional[str] = None
    revenue_streams: Optional[List[str]] = None
    key_customers: Optional[List[str]] = None
    suppliers: Optional[List[str]] = None
    
    # Financial Context
    annual_revenue: Optional[float] = None  # in USD
    employee_count: Optional[int] = None
    established_year: Optional[int] = None
    
    # Market Position
    market_share: Optional[float] = None
    competitors: Optional[List[str]] = None
    
    # Analysis Context
    analysis_date: datetime = None
    loan_amount: Optional[float] = None
    loan_purpose: Optional[str] = None
    
    def __post_init__(self):
        if self.analysis_date is None:
            self.analysis_date = datetime.now()
    
    def get_search_keywords(self) -> List[str]:
        """Generate search keywords for news analysis"""
        keywords = []
        
        # Company name variations
        keywords.append(self.company_name)
        keywords.extend(self.main_products_services)
        keywords.extend(self.target_markets)
        
        # Industry-related terms
        keywords.append(self.industry)
        
        # Market-specific terms
        for market in self.target_markets:
            keywords.append(f"{market} {self.industry}")
            for product in self.main_products_services:
                keywords.append(f"{market} {product}")
        
        return list(set(keywords))  # Remove duplicates
    
    def get_policy_search_terms(self) -> List[str]:
        """Generate terms for policy and regulation search"""
        terms = []
        
        for market in self.target_markets:
            terms.extend([
                f"{market} {self.industry} policy",
                f"{market} {self.industry} regulation",
                f"{market} {self.industry} tariff",
                f"{market} {self.industry} trade",
                f"{market} {self.industry} subsidy",
                f"{market} {self.industry} tax",
            ])
        
        return terms
    
    def get_market_trend_terms(self) -> List[str]:
        """Generate terms for market trend analysis"""
        terms = []
        
        for market in self.target_markets:
            terms.extend([
                f"{market} {self.industry} market",
                f"{market} {self.industry} demand",
                f"{market} {self.industry} growth",
                f"{market} {self.industry} forecast",
                f"{market} {self.industry} outlook",
            ])
        
        return terms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "company_name": self.company_name,
            "industry": self.industry,
            "main_products_services": self.main_products_services,
            "target_markets": self.target_markets,
            "business_model": self.business_model,
            "revenue_streams": self.revenue_streams,
            "key_customers": self.key_customers,
            "suppliers": self.suppliers,
            "annual_revenue": self.annual_revenue,
            "employee_count": self.employee_count,
            "established_year": self.established_year,
            "market_share": self.market_share,
            "competitors": self.competitors,
            "analysis_date": self.analysis_date.isoformat() if self.analysis_date else None,
            "loan_amount": self.loan_amount,
            "loan_purpose": self.loan_purpose,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompanyProfile:
        """Create from dictionary"""
        if data.get("analysis_date"):
            data["analysis_date"] = datetime.fromisoformat(data["analysis_date"])
        return cls(**data)


def create_sample_company() -> CompanyProfile:
    """Create a sample company profile for testing"""
    return CompanyProfile(
        company_name="Samsung Electronics",
        industry="semiconductor",
        main_products_services=["memory chips", "system chips", "foundry services"],
        target_markets=["US", "China", "EU", "Japan"],
        business_model="B2B manufacturing and sales",
        annual_revenue=200000000000,  # $200B
        employee_count=267000,
        established_year=1969,
        market_share=15.2,  # Global semiconductor market
        competitors=["TSMC", "Intel", "SK Hynix", "Micron"],
        loan_amount=5000000000,  # $5B
        loan_purpose="Facility expansion and R&D investment"
    )

