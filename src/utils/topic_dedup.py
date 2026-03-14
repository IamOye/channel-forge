"""
topic_dedup.py — Upload-history duplicate detection for topics.

Normalises topic strings (lowercase, strip punctuation, remove stop words)
then uses difflib.SequenceMatcher to detect near-duplicates.

Usage:
    from src.utils.topic_dedup import filter_new_topics, is_duplicate

    uploaded = ["why saving money keeps you broke"]
    candidates = ["why saving cash keeps you poor", "the debt trap nobody talks about"]
    fresh = filter_new_topics(candidates, uploaded)   # returns second topic only
"""

import difflib
import re

# ---------------------------------------------------------------------------
# Stop words stripped before comparison
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the",
    "why", "how", "what", "when", "where", "who", "which",
    "your", "you", "my", "me", "our", "we", "i",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "this", "that", "these", "those",
    "and", "or", "but", "not", "no",
    "it", "its", "all", "so",
})

# Topics with normalised similarity above this threshold are considered duplicates
SIMILARITY_THRESHOLD: float = 0.70


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_topic(text: str) -> str:
    """
    Normalise a topic string for comparison.

    Steps:
      1. Lowercase
      2. Strip punctuation (keep letters, digits, spaces)
      3. Remove stop words
      4. Collapse whitespace

    Returns:
        Normalised string (may be empty for very short inputs).
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)           # punctuation → space
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def topic_similarity(a: str, b: str) -> float:
    """
    Compute similarity (0.0–1.0) between two topic strings.

    Both strings are normalised before comparison.
    Uses difflib.SequenceMatcher on the full normalised strings.
    """
    na = normalize_topic(a)
    nb = normalize_topic(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def is_duplicate(
    topic: str,
    uploaded_topics: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """
    Return True if *topic* is similar (>= threshold) to any already-uploaded topic.

    Args:
        topic: Candidate topic to check.
        uploaded_topics: List of previously uploaded topic strings.
        threshold: Similarity threshold (0–1, default 0.70).

    Returns:
        True if a duplicate is found; False otherwise.
    """
    norm = normalize_topic(topic)
    for uploaded in uploaded_topics:
        sim = difflib.SequenceMatcher(None, norm, normalize_topic(uploaded)).ratio()
        if sim >= threshold:
            return True
    return False


def filter_new_topics(
    candidates: list[str],
    uploaded_topics: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[str]:
    """
    Return only the candidates that are NOT duplicates of any uploaded topic.

    Args:
        candidates: Topics to filter.
        uploaded_topics: Already-uploaded topic strings.
        threshold: Similarity cutoff.

    Returns:
        Subset of candidates that are sufficiently novel.
    """
    result: list[str] = []
    seen_norms: list[str] = [normalize_topic(u) for u in uploaded_topics]

    for topic in candidates:
        norm = normalize_topic(topic)
        is_dup = any(
            difflib.SequenceMatcher(None, norm, existing).ratio() >= threshold
            for existing in seen_norms
        )
        if not is_dup:
            result.append(topic)
            # Also add to seen so we don't return near-duplicates within candidates
            seen_norms.append(norm)

    return result
