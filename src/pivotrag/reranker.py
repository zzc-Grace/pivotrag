"""Optional HTTP reranker for OpenAI-compatible or similar services."""

from __future__ import annotations

import asyncio
import logging

import httpx
import numpy as np

from pivotrag.config import (
    K3_FINAL,
    NO_RERANK_VECTOR_K,
    RERANK_API_KEY,
    RERANK_BASE_URL,
    RERANK_MODEL,
    USE_RERANKER,
)
from pivotrag.models import PivoGraph, RankedChunk, RetrievalResult

logger = logging.getLogger(__name__)


class PivoReranker:
    """Cross-encoder reranker with an embedding-only selection mode."""

    def __init__(
        self,
        api_key: str = RERANK_API_KEY,
        base_url: str = RERANK_BASE_URL,
        model: str = RERANK_MODEL,
        enabled: bool = USE_RERANKER,
        no_rerank_vector_k: int = NO_RERANK_VECTOR_K,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._enabled = enabled
        self._no_rerank_vector_k = max(0, no_rerank_vector_k)
        self._client: httpx.AsyncClient | None = None

    async def rerank(
        self,
        query: str,
        result: RetrievalResult,
        graph: PivoGraph,
        k3: int = K3_FINAL,
    ) -> list[RankedChunk]:
        """融合双路候选并重排.

        Steps:
        1. 合并 + 按 chunk_id 去重 (保留最高 score)
        2. 构建 (query, chunk_text) 对
        3. 调用 rerank 打分
        4. 按分数降序取 top-k3
        """
        if not self._enabled:
            logger.info(
                "Reranker disabled; selecting %d vector chunks and filling from graph candidates",
                min(self._no_rerank_vector_k, k3),
            )
            return self._select_without_reranker(result, graph, k3)

        if not self._base_url:
            logger.warning(
                "Reranker is enabled but PIVORAG_RERANK_BASE_URL is unset; "
                "falling back to vector + graph selection."
            )
            return self._select_without_reranker(result, graph, k3)

        # 1. Merge and deduplicate
        all_chunks: dict[str, float] = {}
        for c in result.graph_candidates + result.vector_candidates:
            if c.chunk_id not in all_chunks or c.score > all_chunks[c.chunk_id]:
                all_chunks[c.chunk_id] = c.score

        # 2. Gather chunk texts
        documents: list[str] = []
        chunk_ids: list[str] = []
        for cid in all_chunks:
            if cid in graph.chunks:
                documents.append(graph.chunks[cid].text)
                chunk_ids.append(cid)

        if not documents:
            return []

        # 3. Call rerank API with retry
        ranked = await self._call_rerank_api_with_retry(query, documents)
        if ranked is None:
            return self._select_without_reranker(result, graph, k3)

        # 4. Sort and return top-k3
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [
            RankedChunk(
                chunk_id=chunk_ids[idx],
                chunk_text=documents[idx],
                rerank_score=score,
            )
            for idx, score in ranked[:k3]
        ]

    def _select_without_reranker(
        self,
        result: RetrievalResult,
        graph: PivoGraph,
        k3: int,
    ) -> list[RankedChunk]:
        """Select vector top-k, then fill with dense-ranked graph candidates."""
        if k3 <= 0:
            return []

        vector_quota = min(self._no_rerank_vector_k, k3)
        selected: list[tuple[str, float]] = []
        selected_ids: set[str] = set()

        # Keep the configured number of global vector-search results.
        for candidate in result.vector_candidates:
            if len(selected) >= vector_quota:
                break
            if candidate.chunk_id in selected_ids or candidate.chunk_id not in graph.chunks:
                continue
            selected.append((candidate.chunk_id, candidate.score))
            selected_ids.add(candidate.chunk_id)

        graph_scores: dict[str, float] = {}
        if result.query_embedding is None:
            logger.warning(
                "Query embedding is unavailable; ordering graph candidates by retrieval score"
            )

        for candidate in result.graph_candidates:
            cid = candidate.chunk_id
            if cid in selected_ids or cid not in graph.chunks:
                continue

            score = candidate.score
            if result.query_embedding is not None:
                chunk_embedding = graph.chunks[cid].embedding
                dense_score = self._cosine_similarity(
                    result.query_embedding, chunk_embedding
                )
                if dense_score is None:
                    continue
                score = dense_score

            if cid not in graph_scores or score > graph_scores[cid]:
                graph_scores[cid] = score

        graph_needed = k3 - len(selected)
        ranked_graph = sorted(
            graph_scores.items(), key=lambda item: item[1], reverse=True
        )
        for cid, score in ranked_graph[:graph_needed]:
            selected.append((cid, score))
            selected_ids.add(cid)

        # Normally the graph path supplies enough chunks. If it does not, use
        # the remaining vector candidates so the caller can still receive k3.
        if len(selected) < k3:
            for candidate in result.vector_candidates:
                if len(selected) >= k3:
                    break
                if (
                    candidate.chunk_id in selected_ids
                    or candidate.chunk_id not in graph.chunks
                ):
                    continue
                selected.append((candidate.chunk_id, candidate.score))
                selected_ids.add(candidate.chunk_id)

        # Both vector scores and graph scores are cosine similarities whenever
        # the query embedding is present, so they can share one final order.
        if result.query_embedding is not None:
            selected.sort(key=lambda item: item[1], reverse=True)

        return [
            RankedChunk(
                chunk_id=cid,
                chunk_text=graph.chunks[cid].text,
                rerank_score=score,
            )
            for cid, score in selected[:k3]
        ]

    @staticmethod
    def _cosine_similarity(
        query_embedding: np.ndarray,
        chunk_embedding: np.ndarray | None,
    ) -> float | None:
        if chunk_embedding is None:
            return None

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        chunk = np.asarray(chunk_embedding, dtype=np.float32).reshape(-1)
        if query.shape != chunk.shape:
            return None

        denominator = float(np.linalg.norm(query) * np.linalg.norm(chunk))
        if denominator <= 0.0:
            return None
        return float(np.dot(query, chunk) / denominator)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def _call_rerank_api_with_retry(
        self,
        query: str,
        documents: list[str],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> list[tuple[int, float]] | None:
        """Call rerank API with retry mechanism.

        Args:
            query: The query string
            documents: List of documents to rerank
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            List of (index, score) tuples, or None if all retries failed
        """
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                logger.info(f"[Rerank API] Attempt {attempt + 1}/{max_retries}")
                return await self._call_rerank_api(query, documents)
            except httpx.HTTPStatusError as e:
                last_exception = e
                logger.error(f"[Rerank API] HTTP {e.response.status_code} error on attempt {attempt + 1}")
                logger.error(f"[Rerank API] Error response: {e.response.text}")
                # Retry on server errors (5xx) and rate limit (429)
                if e.response.status_code in (429, 500, 502, 503, 504):
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"[Rerank API] Retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                # For other HTTP errors (like 400), don't retry
                logger.error(f"[Rerank API] Non-retryable HTTP error {e.response.status_code}, giving up")
                break
            except Exception as e:
                last_exception = e
                logger.error(f"[Rerank API] Exception on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[Rerank API] Retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"[Rerank API] Failed after {max_retries} attempts")

        logger.warning(
            "Rerank API failed, falling back to vector + graph selection",
            exc_info=last_exception,
        )
        return None

    async def _call_rerank_api(
        self,
        query: str,
        documents: list[str],
    ) -> list[tuple[int, float]]:
        """Call rerank API, returns [(index, score), ...]."""
        client = await self._get_client()

        # Keep requests bounded for providers with document-count limits.
        MAX_DOCUMENTS = 300
        if len(documents) > MAX_DOCUMENTS:
            logger.warning(f"[Rerank API] Documents count ({len(documents)}) exceeds limit ({MAX_DOCUMENTS}), truncating")
            documents = documents[:MAX_DOCUMENTS]

        # Use the common query/documents payload accepted by many rerank APIs.
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }

        logger.info(
            "[Rerank API] Requesting model=%s documents=%d",
            self._model,
            len(documents),
        )

        response = await client.post(
            f"{self._base_url}",
            json=payload,
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self._api_key.strip()}"}
                    if self._api_key.strip()
                    else {}
                ),
            },
        )

        logger.info("[Rerank API] Response status: %d", response.status_code)

        if response.status_code >= 400:
            # Do not log headers, payloads, or response bodies: they may contain
            # credentials, user queries, or provider-specific private details.
            logger.error(
                "[Rerank API] Request failed with HTTP %d",
                response.status_code,
            )

        response.raise_for_status()
        data = response.json()

        # Accept the two common response envelope names.
        results = data.get("results", [])
        if not results and "data" in data:
            # Some services return the payload under ``data`` instead.
            results = data.get("data", [])

        logger.info(f"[Rerank API] Parsed {len(results)} results from response")

        parsed_results = []
        for item in results:
            index = item.get("index")
            # Try different score field names
            score = item.get("relevance_score") or item.get("score") or item.get("relevance")
            if index is not None and score is not None:
                parsed_results.append((index, score))

        logger.info(f"[Rerank API] Successfully parsed {len(parsed_results)} valid results")
        return parsed_results
