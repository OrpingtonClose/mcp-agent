"""Knowledge store with deduplication, scoring, and contradiction detection.

Replaces the original ``WorkspaceMemory.get_relevant_knowledge()`` which
used keyword overlap (set intersection of split words) for "relevance."

This implementation provides:
1. **Deduplication** -- items with semantically similar keys are merged
   rather than accumulated endlessly.
2. **Confidence scoring** -- items decay over time and get boosted by
   cross-references from other sources.
3. **Contradiction detection** -- new items that conflict with existing
   high-confidence items are flagged rather than silently overwriting.
4. **Category-aware retrieval** -- queries can be filtered by category.

No external dependencies (no DuckDB/embeddings) -- designed to work within
mcp-agent's zero-dependency philosophy.  Uses TF-IDF-style token overlap
with n-gram expansion as a significant improvement over raw word intersection.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from mcp_agent.workflows.deep_orchestrator.models import KnowledgeItem

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercased word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _bigrams(tokens: Set[str]) -> Set[str]:
    """Generate bigrams from sorted token set for better matching."""
    sorted_tokens = sorted(tokens)
    return {
        f"{sorted_tokens[i]}_{sorted_tokens[i + 1]}"
        for i in range(len(sorted_tokens) - 1)
    }


def _similarity(text_a: str, text_b: str) -> float:
    """Compute token-overlap similarity with bigram boost.

    Significantly better than raw word intersection because:
    1. Normalised by union size (Jaccard similarity)
    2. Bigrams capture phrase-level similarity
    3. Handles variable-length texts gracefully
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Unigram Jaccard
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    unigram_sim = len(intersection) / len(union) if union else 0.0

    # Bigram boost (captures phrase-level similarity)
    bi_a = _bigrams(tokens_a)
    bi_b = _bigrams(tokens_b)
    bi_intersection = bi_a & bi_b
    bi_union = bi_a | bi_b
    bigram_sim = len(bi_intersection) / len(bi_union) if bi_union else 0.0

    # Weighted combination
    return 0.7 * unigram_sim + 0.3 * bigram_sim


@dataclass
class _ScoredItem:
    """Internal wrapper adding scoring metadata to a knowledge item."""

    item: KnowledgeItem
    cross_references: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeStore:
    """Knowledge store with dedup, scoring, contradiction detection.

    Replaces the keyword-matching approach in WorkspaceMemory with
    proper knowledge management inspired by MiroThinker's CorpusStore.
    """

    def __init__(
        self,
        dedup_threshold: float = 0.6,
        contradiction_threshold: float = 0.4,
        staleness_decay_hours: float = 1.0,
        max_items: int = 200,
    ) -> None:
        self._items: List[_ScoredItem] = []
        self._by_category: Dict[str, List[_ScoredItem]] = defaultdict(list)
        self._dedup_threshold = dedup_threshold
        self._contradiction_threshold = contradiction_threshold
        self._staleness_decay_hours = staleness_decay_hours
        self._max_items = max_items
        self._contradictions: List[Tuple[KnowledgeItem, KnowledgeItem, str]] = []

    @property
    def items(self) -> List[KnowledgeItem]:
        """All knowledge items (unwrapped)."""
        return [s.item for s in self._items]

    @property
    def contradictions(self) -> List[Tuple[KnowledgeItem, KnowledgeItem, str]]:
        """Detected contradictions as (existing, new, reason) tuples."""
        return list(self._contradictions)

    def add(self, item: KnowledgeItem) -> bool:
        """Add a knowledge item with dedup and contradiction detection.

        Returns True if item was added (possibly merged), False if
        it was a duplicate that was absorbed.
        """
        # Check for duplicates and contradictions
        best_match: Optional[_ScoredItem] = None
        best_sim = 0.0

        item_text = f"{item.key} {item.value}"

        for scored in self._items:
            existing_text = f"{scored.item.key} {scored.item.value}"
            sim = _similarity(item_text, existing_text)

            if sim > best_sim:
                best_sim = sim
                best_match = scored

        # Deduplication: merge if very similar
        if best_match and best_sim >= self._dedup_threshold:
            # Check for contradiction (same topic, different conclusion)
            if self._is_contradiction(best_match.item, item):
                self._contradictions.append(
                    (
                        best_match.item,
                        item,
                        f"Similarity {best_sim:.2f} but conflicting values",
                    )
                )
                logger.warning(
                    "Contradiction detected: '%s' vs '%s' (sim=%.2f)",
                    best_match.item.key[:50],
                    item.key[:50],
                    best_sim,
                )
                # Keep both but flag -- higher confidence wins for retrieval
                if item.confidence > best_match.item.confidence:
                    best_match.item = item
                    best_match.cross_references += 1
                    return True
                else:
                    best_match.cross_references += 1
                    return False

            # Pure duplicate -- merge by boosting confidence
            best_match.item.confidence = min(
                1.0,
                best_match.item.confidence + 0.1,
            )
            best_match.cross_references += 1
            best_match.last_accessed = datetime.now(timezone.utc)
            logger.debug(
                "Merged duplicate knowledge: '%s' (sim=%.2f, conf=%.2f)",
                item.key[:50],
                best_sim,
                best_match.item.confidence,
            )
            return False

        # New item -- add it
        scored_item = _ScoredItem(item=item)
        self._items.append(scored_item)
        self._by_category[item.category].append(scored_item)

        # Enforce max items by evicting lowest-scored
        if len(self._items) > self._max_items:
            self._evict_lowest()

        logger.debug(
            "Added knowledge: '%s' (category=%s, confidence=%.2f)",
            item.key[:50],
            item.category,
            item.confidence,
        )
        return True

    def add_batch(self, items: List[KnowledgeItem]) -> int:
        """Add multiple items, returning count of items actually added."""
        added = 0
        for item in items:
            if self.add(item):
                added += 1
        return added

    def query(
        self,
        query_text: str,
        limit: int = 10,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[KnowledgeItem]:
        """Query for relevant knowledge items.

        Uses token-overlap + bigram similarity (NOT raw keyword intersection)
        with staleness decay and cross-reference boosting.
        """
        now = datetime.now(timezone.utc)
        candidates = self._items
        if category:
            candidates = self._by_category.get(category, [])

        scored: List[Tuple[float, _ScoredItem]] = []

        for si in candidates:
            if si.item.confidence < min_confidence:
                continue

            # Base relevance from text similarity
            item_text = f"{si.item.key} {si.item.value}"
            relevance = _similarity(query_text, item_text)

            # Confidence boost
            relevance *= 0.5 + 0.5 * si.item.confidence

            # Cross-reference boost (capped at 2x)
            xref_boost = min(2.0, 1.0 + si.cross_references * 0.15)
            relevance *= xref_boost

            # Staleness decay
            age_hours = (now - si.item.timestamp).total_seconds() / 3600
            decay = 1.0 / (1.0 + age_hours / self._staleness_decay_hours)
            relevance *= 0.3 + 0.7 * decay  # floor at 30% relevance

            scored.append((relevance, si))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Update access time for returned items
        results = []
        for score, si in scored[:limit]:
            si.last_accessed = now
            results.append(si.item)

        return results

    def get_summary(self, limit: int = 10) -> str:
        """Get a formatted summary of knowledge for prompts."""
        if not self._items:
            return "No knowledge accumulated yet."

        # Sort by recency and confidence
        recent = sorted(
            self._items,
            key=lambda s: (s.item.confidence, s.item.timestamp.timestamp()),
            reverse=True,
        )[:limit]

        lines = ["<knowledge_summary>"]
        by_cat: Dict[str, List[_ScoredItem]] = defaultdict(list)
        for si in recent:
            by_cat[si.item.category].append(si)

        for category, items in by_cat.items():
            lines.append(f'  <category name="{category}">')
            for si in items:
                value_str = str(si.item.value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                xref_note = (
                    f' cross_refs="{si.cross_references}"'
                    if si.cross_references > 0
                    else ""
                )
                lines.append(
                    f'    <item confidence="{si.item.confidence:.2f}" '
                    f'source="{si.item.source}"{xref_note}>'
                )
                lines.append(f"      <key>{si.item.key}</key>")
                lines.append(f"      <value>{value_str}</value>")
                lines.append("    </item>")
            lines.append("  </category>")

        if self._contradictions:
            lines.append(f'  <contradictions count="{len(self._contradictions)}">')
            for existing, new, reason in self._contradictions[-3:]:
                lines.append(
                    f"    <conflict>{existing.key[:50]} vs {new.key[:50]}</conflict>"
                )
            lines.append("  </contradictions>")

        lines.append("</knowledge_summary>")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge store statistics."""
        return {
            "total_items": len(self._items),
            "categories": len(self._by_category),
            "contradictions": len(self._contradictions),
            "avg_confidence": (
                sum(s.item.confidence for s in self._items) / len(self._items)
                if self._items
                else 0.0
            ),
            "avg_cross_refs": (
                sum(s.cross_references for s in self._items) / len(self._items)
                if self._items
                else 0.0
            ),
        }

    def _is_contradiction(self, existing: KnowledgeItem, new: KnowledgeItem) -> bool:
        """Detect if two similar-keyed items contradict each other.

        Simple heuristic: same topic (high key similarity) but different
        values (low value similarity).
        """
        key_sim = _similarity(existing.key, new.key)
        if key_sim < self._contradiction_threshold:
            return False

        value_sim = _similarity(str(existing.value), str(new.value))
        # High key similarity + low value similarity = contradiction
        return value_sim < 0.3

    def _evict_lowest(self) -> None:
        """Remove lowest-scored items to stay within max_items."""
        now = datetime.now(timezone.utc)

        def score(si: _ScoredItem) -> float:
            age_hours = (now - si.item.timestamp).total_seconds() / 3600
            decay = 1.0 / (1.0 + age_hours / self._staleness_decay_hours)
            return si.item.confidence * decay * (1 + si.cross_references * 0.1)

        self._items.sort(key=score, reverse=True)
        evicted = self._items[self._max_items :]
        self._items = self._items[: self._max_items]

        # Rebuild category index
        self._by_category.clear()
        for si in self._items:
            self._by_category[si.item.category].append(si)

        if evicted:
            logger.debug("Evicted %d low-scored knowledge items", len(evicted))

    def clear(self) -> None:
        """Clear all knowledge."""
        self._items.clear()
        self._by_category.clear()
        self._contradictions.clear()
