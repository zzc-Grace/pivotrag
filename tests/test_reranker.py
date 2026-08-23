import numpy as np
import pytest

from pivotrag.models import (
    ChunkNode,
    PivoGraph,
    RetrievalCandidate,
    RetrievalResult,
)
from pivotrag.reranker import PivoReranker


def _graph(embeddings: dict[str, list[float]]) -> PivoGraph:
    return PivoGraph(
        entities={},
        chunks={
            cid: ChunkNode(
                id=cid,
                text=f"text-{cid}",
                embedding=np.array(embedding, dtype=np.float32),
            )
            for cid, embedding in embeddings.items()
        },
        edges=[],
    )


@pytest.mark.asyncio
async def test_disabled_reranker_uses_vector_quota_and_dense_graph_fill() -> None:
    graph = _graph(
        {
            "v1": [1.0, 0.0],
            "v2": [0.95, 0.05],
            "overlap": [1.0, 0.0],
            "g_high": [1.0, 0.0],
            "g_mid": [0.8, 0.6],
            "g_low": [0.0, 1.0],
        }
    )
    result = RetrievalResult(
        graph_candidates=[
            RetrievalCandidate("g_low", 100.0, "graph"),
            RetrievalCandidate("overlap", 90.0, "graph"),
            RetrievalCandidate("g_mid", 1.0, "graph"),
            RetrievalCandidate("g_high", 0.1, "graph"),
        ],
        vector_candidates=[
            RetrievalCandidate("overlap", 1.0, "vector"),
            RetrievalCandidate("v1", 0.99, "vector"),
            RetrievalCandidate("v2", 0.98, "vector"),
        ],
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    ranked = await PivoReranker(
        enabled=False, no_rerank_vector_k=2
    ).rerank("query", result, graph, k3=4)

    assert {item.chunk_id for item in ranked} == {
        "overlap",
        "v1",
        "g_high",
        "g_mid",
    }
    assert len({item.chunk_id for item in ranked}) == 4


@pytest.mark.asyncio
async def test_disabled_reranker_uses_vector_reserve_when_graph_is_short() -> None:
    graph = _graph(
        {
            "v1": [1.0, 0.0],
            "v2": [0.9, 0.1],
            "v3": [0.8, 0.2],
            "v4": [0.7, 0.3],
        }
    )
    result = RetrievalResult(
        graph_candidates=[RetrievalCandidate("v1", 1.0, "graph")],
        vector_candidates=[
            RetrievalCandidate("v1", 1.0, "vector"),
            RetrievalCandidate("v2", 0.9, "vector"),
            RetrievalCandidate("v3", 0.8, "vector"),
            RetrievalCandidate("v4", 0.7, "vector"),
        ],
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
    )

    ranked = await PivoReranker(
        enabled=False, no_rerank_vector_k=2
    ).rerank("query", result, graph, k3=4)

    assert {item.chunk_id for item in ranked} == {"v1", "v2", "v3", "v4"}
    assert len(ranked) == 4
