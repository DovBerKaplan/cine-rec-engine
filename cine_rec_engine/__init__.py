"""cine-rec-engine — content-based movie & series recommendations from your own PostgreSQL."""

from .service import RecommendationService

__version__ = "0.5.2"
__all__ = ["RecommendationService", "__version__"]
