"""spaCy-based entity extraction from text chunks.

Uses a local spaCy model (configured via SPACY_MODEL_PATH) to extract named
entities during knowledge-base construction. This avoids calling an LLM for
entity extraction at build time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pivotrag.config import (
    SPACY_BATCH_SIZE,
    SPACY_ENTITY_TYPES,
    SPACY_MODEL_PATH,
)

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)

# Lazy singleton for the spaCy nlp object.
_nlp: "spacy.Language | None" = None

_LABEL_TO_DESCRIPTION: dict[str, str] = {
    "PERSON": "a person",
    "ORG": "an organization",
    "GPE": "a geopolitical entity",
    "LOC": "a location",
    "FAC": "a facility",
    "PRODUCT": "a product",
    "EVENT": "an event",
    "WORK_OF_ART": "a work of art",
    "LAW": "a law",
    "LANGUAGE": "a language",
    "NORP": "a nationality or religious/political group",
}


def _load_nlp() -> "spacy.Language":
    """Load the configured spaCy model lazily."""
    global _nlp
    if _nlp is None:
        import spacy as _spacy

        logger.info("Loading spaCy model from %s", SPACY_MODEL_PATH)
        try:
            _nlp = _spacy.load(SPACY_MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load spaCy model from {SPACY_MODEL_PATH!r}. "
                "If this is a transformer model (e.g. en_core_web_trf), "
                "ensure 'spacy-transformers' and 'torch' are installed."
            ) from exc
    return _nlp


def _description_for_label(label: str) -> str:
    """Return a short description for a spaCy entity label."""
    return _LABEL_TO_DESCRIPTION.get(label, f"a {label.lower()} entity")


def extract_entities_from_chunks(
    chunks_data: list[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Synchronous spaCy entity extraction over a list of chunks.

    Args:
        chunks_data: [(chunk_id, chunk_text), ...]

    Returns:
        {chunk_id: [(entity_name, description), ...]}
    """
    nlp = _load_nlp()
    texts = [text for _, text in chunks_data]
    results: dict[str, list[tuple[str, str]]] = {}

    for (chunk_id, _), doc in zip(
        chunks_data, nlp.pipe(texts, batch_size=SPACY_BATCH_SIZE)
    ):
        seen: set[str] = set()
        entities: list[tuple[str, str]] = []
        for ent in doc.ents:
            if ent.label_ not in SPACY_ENTITY_TYPES:
                continue
            key = ent.text.lower()
            if key in seen:
                continue
            seen.add(key)
            description = _description_for_label(ent.label_)
            entities.append((ent.text, description))
        results[chunk_id] = entities

    return results


async def batch_extract_entities(
    chunks_data: list[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Async wrapper around spaCy extraction.

    spaCy is CPU-bound, so run it in the default executor to avoid blocking the
    event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_entities_from_chunks, chunks_data)
