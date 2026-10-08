"""cine-rec-engine — content-based movie & series recommendations from your own PostgreSQL."""

from .service import RecommendationService

__version__ = "0.11.0"
__all__ = ["RecommendationService", "__version__"]
