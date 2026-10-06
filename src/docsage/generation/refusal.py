"""F5: refuse with "not in these documents" when the best evidence is too weak."""
from docsage.config import settings
from docsage.models import Hit

REFUSAL_TEXT = "Not in these documents."


def should_refuse(hits: list[Hit], threshold: float | None = None) -> bool:
    """Uses the best vector cosine similarity, which is comparable across queries
    (RRF and re-ranker scores are not)."""
    threshold = settings.refusal_threshold if threshold is None else threshold
    return not hits or max(h.vector_score for h in hits) < threshold
