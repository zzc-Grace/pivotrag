"""Graph persistence: PivoGraph <-> Parquet + npy."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pivotrag.models import ChunkNode, Edge, EntityNode, PivoGraph

logger = logging.getLogger(__name__)

NODES_FILE = "pivorag_nodes.parquet"
EDGES_FILE = "pivorag_edges.parquet"
ENTITY_EMBS_FILE = "pivorag_entity_embs.npy"
CHUNK_EMBS_FILE = "pivorag_chunk_embs.npy"


def save_graph(graph: PivoGraph, output_dir: str) -> None:
    """保存图到 Parquet + npy 文件."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # ── Nodes ──
    entity_embs: list[np.ndarray] = []
    entity_rows: list[dict] = []
    chunk_embs: list[np.ndarray] = []
    chunk_rows: list[dict] = []
    emb_idx = 0
    chunk_emb_idx = 0

    for entity in graph.entities.values():
        if entity.embedding is not None:
            entity_embs.append(entity.embedding)
            eidx = emb_idx
            emb_idx += 1
        else:
            eidx = -1
        entity_rows.append(
            {
                "id": entity.id,
                "type": "entity",
                "name": entity.name,
                "description": entity.description,
                "source_chunk_ids": json.dumps(entity.source_chunk_ids),
                "embedding_index": eidx,
                "text": "",
                "n_tokens": 0,
                "document_ids": "",
            }
        )

    for chunk in graph.chunks.values():
        if chunk.embedding is not None:
            chunk_embs.append(chunk.embedding)
            cidx = chunk_emb_idx
            chunk_emb_idx += 1
        else:
            cidx = -1
        chunk_rows.append(
            {
                "id": chunk.id,
                "type": "chunk",
                "name": "",
                "description": "",
                "source_chunk_ids": "",
                "embedding_index": cidx,
                "text": chunk.text,
                "n_tokens": chunk.n_tokens,
                "document_ids": json.dumps(chunk.document_ids),
            }
        )

    nodes_df = pd.DataFrame(entity_rows + chunk_rows)
    nodes_df.to_parquet(path / NODES_FILE, index=False)

    # ── Edges ──
    edges_df = pd.DataFrame(
        [
            {
                "source": e.source,
                "target": e.target,
                "type": e.type,
                "weight": e.weight,
            }
            for e in graph.edges
        ]
    )
    edges_df.to_parquet(path / EDGES_FILE, index=False)

    # ── Embeddings ──
    if entity_embs:
        np.save(path / ENTITY_EMBS_FILE, np.stack(entity_embs))
    if chunk_embs:
        np.save(path / CHUNK_EMBS_FILE, np.stack(chunk_embs))

    logger.info(
        "Graph saved to %s: %d nodes, %d edges",
        output_dir,
        len(entity_rows) + len(chunk_rows),
        len(graph.edges),
    )


def load_graph(output_dir: str) -> PivoGraph:
    """从 Parquet + npy 加载完整图."""
    path = Path(output_dir)

    nodes_df = pd.read_parquet(path / NODES_FILE)
    edges_df = pd.read_parquet(path / EDGES_FILE)

    entity_embs_path = path / ENTITY_EMBS_FILE
    chunk_embs_path = path / CHUNK_EMBS_FILE
    entity_embs = np.load(entity_embs_path) if entity_embs_path.exists() else None
    chunk_embs = np.load(chunk_embs_path) if chunk_embs_path.exists() else None

    entities: dict[str, EntityNode] = {}
    chunks: dict[str, ChunkNode] = {}

    for _, row in nodes_df.iterrows():
        if row["type"] == "entity":
            emb: np.ndarray | None = None
            if entity_embs is not None and row["embedding_index"] >= 0:
                emb = entity_embs[int(row["embedding_index"])]
            entities[row["id"]] = EntityNode(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                embedding=emb,
                source_chunk_ids=(
                    json.loads(row["source_chunk_ids"])
                    if row["source_chunk_ids"]
                    else []
                ),
            )
        elif row["type"] == "chunk":
            cemb: np.ndarray | None = None
            if chunk_embs is not None and row["embedding_index"] >= 0:
                cemb = chunk_embs[int(row["embedding_index"])]
            chunks[row["id"]] = ChunkNode(
                id=row["id"],
                text=row["text"],
                embedding=cemb,
                n_tokens=int(row.get("n_tokens", 0)),
                document_ids=(
                    json.loads(row["document_ids"])
                    if row.get("document_ids")
                    else []
                ),
            )

    edges: list[Edge] = [
        Edge(
            source=r["source"],
            target=r["target"],
            type=r["type"],
            weight=float(r["weight"]),
        )
        for _, r in edges_df.iterrows()
    ]

    logger.info(
        "Graph loaded: %d entities, %d chunks, %d edges",
        len(entities),
        len(chunks),
        len(edges),
    )
    return PivoGraph(entities=entities, chunks=chunks, edges=edges)


def graph_exists(output_dir: str) -> bool:
    """检查图文件是否已存在."""
    path = Path(output_dir)
    return (path / NODES_FILE).exists() and (path / EDGES_FILE).exists()
