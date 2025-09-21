from .openrouter_impact_scorer import OpenRouterImpactScorer, ImpactScore

# Alias for backward compatibility
ImpactScorer = OpenRouterImpactScorer

__all__ = ["ImpactScorer", "OpenRouterImpactScorer", "ImpactScore"]

