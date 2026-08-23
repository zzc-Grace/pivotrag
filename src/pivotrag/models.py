"""Shared data structures for PivoRAG modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class EntityNode:
    """实体节点."""

    id: str
    name: str
    description: str
    embedding: np.ndarray | None = field(default=None, repr=False)
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ChunkNode:
    """文本块节点."""

    id: str
    text: str
    embedding: np.ndarray | None = field(default=None, repr=False)
    n_tokens: int = 0
    document_ids: list[str] = field(default_factory=list)


EdgeType = Literal["adjacency", "similarity", "belongs_to"]


@dataclass
class Edge:
    """图中的一条边."""

    source: str
    target: str
    type: EdgeType
    weight: float = 1.0


@dataclass
class PivoGraph:
    """完整的异构图，graph_builder 的输出、graph_storage 的读写单位."""

    entities: dict[str, EntityNode]
    chunks: dict[str, ChunkNode]
    edges: list[Edge]

    def entity_subgraph_edges(self) -> list[Edge]:
        """返回仅 entity-entity 边 (adjacency + similarity), 供 PPR 使用."""
        return [e for e in self.edges if e.type in ("adjacency", "similarity")]

    def entity_to_chunks(self, entity_id: str) -> list[str]:
        """通过 belongs_to 边获取实体关联的 chunk id 列表."""
        return [
            e.target
            for e in self.edges
            if e.type == "belongs_to" and e.source == entity_id
        ]


@dataclass
class RetrievalCandidate:
    """单路检索的一个候选 chunk."""

    chunk_id: str
    score: float
    source: Literal["graph", "vector"]


@dataclass
class RetrievalResult:
    """双路检索结果及无模型排序可复用的 query embedding."""

    graph_candidates: list[RetrievalCandidate]
    vector_candidates: list[RetrievalCandidate]
    query_embedding: np.ndarray | None = field(default=None, repr=False)


@dataclass
class RankedChunk:
    """Reranker 最终输出."""

    chunk_id: str
    chunk_text: str
    rerank_score: float


@dataclass
class TokenUsage:
    """Token 用量统计."""

    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embedding_input_tokens: int = 0

    def add(self, other: TokenUsage) -> None:
        self.llm_input_tokens += other.llm_input_tokens
        self.llm_output_tokens += other.llm_output_tokens
        self.embedding_input_tokens += other.embedding_input_tokens

    def to_dict(self) -> dict:
        return {
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "embedding_input_tokens": self.embedding_input_tokens,
            "llm_total_tokens": self.llm_input_tokens + self.llm_output_tokens,
            "total_tokens": self.llm_input_tokens + self.llm_output_tokens + self.embedding_input_tokens,
        }
