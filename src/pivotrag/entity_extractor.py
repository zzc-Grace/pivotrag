"""LLM-based entity extraction from text chunks."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from pivotrag.config import (
    ENTITY_DESC_MAX_LEN,
    EXTRACT_MAX_RETRIES,
    MAX_CONCURRENT_LLM_CALLS,
)
from pivotrag.http_client import chat_completion
from pivotrag.models import TokenUsage

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = """
You are a Named Entity Recognition (NER) system.
Extract all explicitly mentioned named entities and assign each a short semantic description.

Description rules:
- Use a concise, generalizable phrase (e.g., "a historical event", "a person's name", "a geographic location").
- Describe what the entity *is*, not what it does.

Extraction rules:
- Entity name must exactly match the surface form in the text.
- When multiple surface forms refer to the same real-world entity (e.g., different translations, abbreviations, language variants), extract ONLY ONE entry using the first or most prominent form. Do NOT create separate entries for variants.
- Extract each unique real-world entity only once.
- Do not include generic nouns, pronouns, or template words.
- Do not extract dates, times, or temporal expressions as entities.

CRITICAL OUTPUT REQUIREMENTS:
1. Output ONLY a valid JSON array. No markdown code blocks, no explanations, no comments.
2. Each entity must be an object with exactly two fields: "name" and "description".

Output format examples:
- Single entity: [{"name": "Eiffel Tower", "description": "a landmark structure in Paris"}]
- Multiple entities: [{"name": "Eiffel Tower", "description": "a landmark structure in Paris"}, {"name": "Paris", "description": "a capital city in France"}]
- No entities: []
"""


async def extract_entities_from_chunk(
    chunk_text: str,
    base_url: str,
    api_key: str,
    model: str,
    usage_counter: TokenUsage | None = None,
) -> list[tuple[str, str]]:
    """从单个 chunk 抽取实体.

    Returns:
        [(entity_name, description), ...]
        description 不超过 ENTITY_DESC_MAX_LEN 字符.
    """
    for attempt in range(EXTRACT_MAX_RETRIES):
        try:
            content, usage = await chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": chunk_text,
                    },
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            if usage_counter and usage:
                usage_counter.llm_input_tokens += usage.get("prompt_tokens", 0)
                usage_counter.llm_output_tokens += usage.get("completion_tokens", 0)
            entities, parsed_ok = _parse_entities(content)
            if not parsed_ok:
                raise ValueError(
                    f"LLM response parsing failed — raw response: {content[:200]!r}"
                )
            if not entities:
                logger.debug("No entities in chunk (parsed OK, genuinely empty)")
            return entities
        except Exception:
            logger.warning(
                "Entity extraction attempt %d failed", attempt + 1, exc_info=True
            )
    logger.error(
        "Failed to extract entities after %d attempts", EXTRACT_MAX_RETRIES
    )
    return []


def _parse_entities(content: str) -> tuple[list[tuple[str, str]], bool]:
    """Parse LLM response into list of (name, description) tuples.

    Returns:
        (entities, parsed_successfully):
        - parsed_successfully=True  → LLM 有效响应（可能是空列表 []）
        - parsed_successfully=False → 无法解析 LLM 输出
    """
    # Clean the content first
    cleaned = content.strip()

    # Strip optional reasoning tags and special tokens emitted by some models.
    cleaned = re.sub(r"<thinking>[^<]*(?:<(?!/thinking>)[^<]*)*</thinking>", "", cleaned)
    cleaned = re.sub(r"<\|.*?\|>", "", cleaned).strip()

    # Remove common prefixes that LLM might add
    cleaned = re.sub(
        r"^\s*(?:Here\s+(?:is|are)\s+the\s+)?(?:JSON\s+array|entities|output|result)s?\s*:\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )

    # 1) Direct JSON parse
    try:
        items = json.loads(cleaned)
        if isinstance(items, list):
            result = _validate_items(items)
            if not result and items:
                logger.debug("JSON parsed OK but all items invalid: %.200s", content)
            return result, True
    except json.JSONDecodeError as e:
        logger.debug("Direct JSON parse failed (%s): %.200s", e, content)

    # 2) Extract from markdown code block (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(1))
            if isinstance(items, list):
                result = _validate_items(items)
                if not result and items:
                    logger.debug("Code block JSON parsed OK but all items invalid: %.200s", content)
                return result, True
        except json.JSONDecodeError as e:
            logger.debug("Code block JSON parse failed (%s): %.200s", e, content)

    # 3) Try to fix common JSON issues (trailing commas, single quotes)
    fixed = re.sub(r",\s*([\]}])", r"\1", cleaned)  # trailing commas
    fixed = fixed.replace("'", '"')  # single quotes → double quotes
    if fixed != cleaned:
        try:
            items = json.loads(fixed)
            if isinstance(items, list):
                result = _validate_items(items)
                if not result and items:
                    logger.debug("Fixed JSON parsed OK but all items invalid: %.200s", content)
                return result, True
        except json.JSONDecodeError as e:
            logger.debug("Fixed JSON parse failed (%s): %.200s", e, content)

    # 4) Regex fallback: extract {"name": "...", "description": "..."} objects individually
    entity_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"description"\s*:\s*"([^"]*)"\s*\}'
    matches = re.findall(entity_pattern, cleaned)
    if matches:
        logger.debug("Regex extracted %d entities from unstructured response", len(matches))
        return [(name, desc) for name, desc in matches], True

    logger.warning("All parse methods failed for response: %.200s", content)
    return [], False


def _validate_items(items: list) -> list[tuple[str, str]]:
    """Validate and truncate entity items."""
    result: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        desc = item.get("description", "").strip()
        if not name:
            continue
        if len(desc) > ENTITY_DESC_MAX_LEN:
            desc = desc[:ENTITY_DESC_MAX_LEN]
        result.append((name, desc))
    return result


async def batch_extract_entities(
    chunks: list[tuple[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    max_concurrency: int = MAX_CONCURRENT_LLM_CALLS,
    usage_counter: TokenUsage | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """批量异步抽取实体.

    Args:
        chunks: [(chunk_id, chunk_text), ...]
        base_url: LLM API base URL
        api_key: LLM API key
        model: 模型名称
        usage_counter: 可选的 token 用量计数器

    Returns:
        {chunk_id: [(name, desc), ...]}
        单个 chunk 失败不阻断整体.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    results: dict[str, list[tuple[str, str]]] = {}

    async def _process(chunk_id: str, chunk_text: str) -> None:
        async with semaphore:
            try:
                entities = await extract_entities_from_chunk(
                    chunk_text, base_url, api_key, model, usage_counter
                )
                results[chunk_id] = entities
            except Exception:
                logger.warning(
                    "Skipping chunk %s due to error", chunk_id, exc_info=True
                )
                results[chunk_id] = []

    tasks = [_process(cid, ctext) for cid, ctext in chunks]
    await asyncio.gather(*tasks)
    return results
