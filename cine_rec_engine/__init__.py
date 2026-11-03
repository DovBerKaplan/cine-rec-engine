"""cine-rec-engine — content-based movie & series recommendations from your own PostgreSQL."""

from .service import RecommendationService

__version__ = "1.1.0"
__all__ = ["RecommendationService", "__version__"]
