"""Lightweight async HTTP helpers for OpenAI-compatible chat/embedding APIs."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# Shared async client (proxy-free, matching reranker pattern)
_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(proxy=None, trust_env=False, timeout=120.0)
    return _client


def _normalize_base_url(base_url: str) -> str:
    """Strip trailing /v1 so we can append consistent endpoint paths."""
    normalized = base_url.strip().rstrip("/").removesuffix("/v1")
    if not normalized:
        raise ValueError(
            "Model service URL is not configured. Set the relevant "
            "PIVORAG_*_BASE_URL environment variable."
        )
    return normalized


def _headers(api_key: str) -> dict[str, str]:
    """Build request headers, allowing local unauthenticated services."""
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


async def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> tuple[str, dict]:
    """POST to /v1/chat/completions, return (content, usage_dict)."""
    client = await _get_client()
    url = f"{_normalize_base_url(base_url)}/v1/chat/completions"
    resp = await client.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        headers=_headers(api_key),
    )
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {})
    return content, usage


async def create_embeddings(
    base_url: str,
    api_key: str,
    model: str,
    input_texts: list[str],
    dimensions: int | None = None,
) -> tuple[list[list[float]], dict]:
    """POST to /v1/embeddings, return (embeddings_list, usage_dict)."""
    client = await _get_client()
    url = f"{_normalize_base_url(base_url)}/v1/embeddings"
    body: dict = {"model": model, "input": input_texts}
    if dimensions is not None:
        body["dimensions"] = dimensions

    resp = await client.post(
        url,
        json=body,
        headers=_headers(api_key),
    )
    resp.raise_for_status()
    data = resp.json()

    # Sort by index to guarantee order matches input_texts
    sorted_items = sorted(data["data"], key=lambda x: x["index"])
    embeddings = [item["embedding"] for item in sorted_items]
    usage = data.get("usage", {})
    return embeddings, usage
