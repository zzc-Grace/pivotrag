"""Graph construction: edges, similarity search, entity merging."""

from __future__ import annotations

import logging
import unicodedata

import faiss
import numpy as np

import math

from pivotrag.config import E_SIM_TOP_L, SIMILARITY_THRESHOLD
from pivotrag.models import ChunkNode, Edge, EntityNode, PivoGraph

logger = logging.getLogger(__name__)


def _normalize_entity_name(name: str) -> str:
    """Normalize an entity surface form for same-name merging."""
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def _make_entity_id(chunk_id: str, index: int) -> str:
    """Generate a stable entity ID."""
    return f"ent_{chunk_id}_{index}"


# ── Adjacency edges ──


def build_adjacency_edges(
    chunk_entities: dict[str, list[tuple[str, str]]],
    entity_idf: dict[str, float] | None = None,
) -> list[Edge]:
    """同一 chunk 内相邻实体对建边, 权重 = IDF(e_i) * IDF(e_j)."""
    edges: list[Edge] = []
    for chunk_id, entities in chunk_entities.items():
        for i in range(len(entities) - 1):
            id_a = _make_entity_id(chunk_id, i)
            id_b = _make_entity_id(chunk_id, i + 1)
            if entity_idf is not None:
                w = entity_idf.get(id_a, 1.0) * entity_idf.get(id_b, 1.0)
            else:
                w = 1.0
            edges.append(
                Edge(source=id_a, target=id_b, type="adjacency", weight=w)
            )
    return edges


# ── Union-Find for transitive merging ──


class _UnionFind:
    """Disjoint-set structure for transitive entity merging."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep lexicographically smaller id as root
            if ra < rb:
                self._parent[rb] = ra
            else:
                self._parent[ra] = rb


# ── Similarity search ──


def find_similar_entities(
    entities: dict[str, EntityNode],
    threshold: float = SIMILARITY_THRESHOLD,
    top_l: int = E_SIM_TOP_L,
    entity_idf: dict[str, float] | None = None,
) -> tuple[list[Edge], dict[str, str]]:
    """跨 chunk 寻找相似实体 (ANN top-L).

    Edge weight = sqrt(sup_i * sup_j) * margin * sqrt(idf_i * idf_j)
    where:
        sup_x = support count (number of FAISS neighbors above threshold)
        margin = cosine - threshold
        idf_x = entity IDF score

    Returns:
        - similarity_edges: 相似边 (cosine > threshold 且 name 不同)
        - merge_map: {old_id -> kept_id} (归一化 name 相同且 cosine > threshold)
    """
    if len(entities) < 2:
        return [], {}

    ids = list(entities.keys())

    # Filter to entities with embeddings
    valid_entries = [
        (eid, entities[eid])
        for eid in ids
        if entities[eid].embedding is not None
    ]
    if len(valid_entries) < 2:
        return [], {}

    valid_ids = [eid for eid, _ in valid_entries]
    normalized_names = {
        eid: _normalize_entity_name(entity.name) for eid, entity in valid_entries
    }
    vecs = np.stack(
        [ent.embedding for _, ent in valid_entries]  # type: ignore[arg-type]
    ).astype(np.float32)
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    k = min(top_l, len(valid_ids) - 1)
    scores, indices = index.search(vecs, k + 1)  # +1 includes self

    # First pass: compute support counts per entity
    support_count: dict[str, int] = {eid: 0 for eid in valid_ids}
    for i, eid in enumerate(valid_ids):
        for j in range(k + 1):
            sim = float(scores[i][j])
            if sim <= threshold:
                continue
            other_idx = int(indices[i][j])
            if other_idx == i or other_idx >= len(valid_ids):
                continue
            support_count[eid] += 1

    similarity_edges: list[Edge] = []
    uf = _UnionFind()

    for i, eid in enumerate(valid_ids):
        for j in range(k + 1):
            sim = float(scores[i][j])
            if sim <= threshold:
                continue
            other_idx = int(indices[i][j])
            if other_idx == i or other_idx >= len(valid_ids):
                continue
            other_id = valid_ids[other_idx]
            if normalized_names[eid] == normalized_names[other_id]:
                # Same name + high similarity → merge (any chunk)
                uf.union(eid, other_id)
            else:
                # Different name + high similarity → edge, regardless of source chunk
                margin = sim - threshold
                sup_factor = math.sqrt(
                    max(support_count[eid], 1) * max(support_count[other_id], 1)
                )
                if entity_idf is not None:
                    idf_factor = math.sqrt(
                        entity_idf.get(eid, 1.0) * entity_idf.get(other_id, 1.0)
                    )
                else:
                    idf_factor = 1.0
                w = sup_factor * margin * idf_factor
                similarity_edges.append(
                    Edge(
                        source=eid,
                        target=other_id,
                        type="similarity",
                        weight=w,
                    )
                )

    # Deduplicate undirected edges
    seen: set[tuple[str, str]] = set()
    deduped: list[Edge] = []
    for e in similarity_edges:
        key = tuple(sorted([e.source, e.target]))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    # Build merge_map from UnionFind
    merge_map: dict[str, str] = {}
    for eid in valid_ids:
        root = uf.find(eid)
        if root != eid:
            merge_map[eid] = root

    return deduped, merge_map


# ── Entity merging ──


def merge_entities(
    entities: dict[str, EntityNode],
    edges: list[Edge],
    merge_map: dict[str, str],
) -> tuple[dict[str, EntityNode], list[Edge]]:
    """根据 merge_map 合并实体节点和关联边."""
    if not merge_map:
        return entities, edges

    # Resolve transitive merges to final target
    resolved: dict[str, str] = {}
    for old_id in merge_map:
        target = merge_map[old_id]
        while target in merge_map:
            target = merge_map[target]
        resolved[old_id] = target

    def _remap(node_id: str) -> str:
        return resolved.get(node_id, node_id)

    # Merge entity nodes
    new_entities: dict[str, EntityNode] = {}
    for eid, entity in entities.items():
        target_id = _remap(eid)
        if target_id == eid:
            new_entities[eid] = entity
        else:
            if target_id not in new_entities:
                new_entities[target_id] = entities[target_id]
            # Merge source_chunk_ids
            new_entities[target_id].source_chunk_ids = list(
                set(
                    new_entities[target_id].source_chunk_ids
                    + entity.source_chunk_ids
                )
            )

    # Remap and deduplicate edges
    edge_best: dict[tuple[str, str, str], float] = {}
    for e in edges:
        src = _remap(e.source)
        tgt = _remap(e.target)
        if src == tgt:
            continue  # Self-loop after merge
        key = (src, tgt, e.type)
        if key not in edge_best or e.weight > edge_best[key]:
            edge_best[key] = e.weight

    new_edges = [
        Edge(source=s, target=t, type=tp, weight=w)
        for (s, t, tp), w in edge_best.items()
    ]

    merged_count = sum(1 for v in resolved.values() if _remap(v) != v)
    logger.info("Merged %d entities into existing ones", merged_count)
    return new_entities, new_edges


# ── Main entry point ──


def _compute_entity_idf(
    chunk_entities: dict[str, list[tuple[str, str]]],
) -> dict[str, float]:
    """Compute IDF for each entity: log(1 + N_chunks / (df(e) + 1))."""
    n_chunks = len(chunk_entities)
    df: dict[str, int] = {}
    for chunk_id, ent_list in chunk_entities.items():
        for idx, _ in enumerate(ent_list):
            eid = _make_entity_id(chunk_id, idx)
            df[eid] = df.get(eid, 0) + 1
    return {eid: math.log(1.0 + n_chunks / (count + 1.0)) for eid, count in df.items()}


def _parse_chunk_index(chunk_id: str) -> int:
    """Extract the numeric index suffix from a chunk ID (e.g. 'doc_3' → 3)."""
    parts = chunk_id.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def build_graph(
    chunk_entities: dict[str, list[tuple[str, str]]],
    chunk_nodes: dict[str, ChunkNode],
    entity_embeddings: dict[str, np.ndarray],
) -> PivoGraph:
    """完整的图构建流程 (纯计算, 无 I/O).

    Steps:
    1. 为每个 chunk 的实体生成 EntityNode
    2. 计算 entity IDF
    3. build_adjacency_edges (IDF 加权)
    4. find_similar_entities → similarity 边 (margin+support+IDF 加权) + merge_map
    5. merge_entities
    6. 添加 belongs_to 边 (IDF + position decay 加权)
    7. 封装为 PivoGraph
    """
    # Step 1: Create EntityNodes
    entities: dict[str, EntityNode] = {}
    for chunk_id, ent_list in chunk_entities.items():
        for idx, (name, desc) in enumerate(ent_list):
            eid = _make_entity_id(chunk_id, idx)
            embedding = entity_embeddings.get(eid)
            entities[eid] = EntityNode(
                id=eid,
                name=name,
                description=desc,
                embedding=embedding,
                source_chunk_ids=[chunk_id],
            )

    # Step 2: Compute entity IDF (pre-merge, from chunk_entities)
    entity_idf = _compute_entity_idf(chunk_entities)

    # Step 3: Adjacency edges (IDF-weighted)
    adjacency_edges = build_adjacency_edges(chunk_entities, entity_idf)

    # Step 4: Similarity search + merge candidates (margin+support+IDF)
    similarity_edges, merge_map = find_similar_entities(
        entities, entity_idf=entity_idf
    )

    # Step 5: Merge entities
    all_edges = adjacency_edges + similarity_edges
    entities, all_edges = merge_entities(entities, all_edges, merge_map)

    # Step 6: belongs_to edges (entity → chunk, IDF * position_decay)
    belongs_to_edges: list[Edge] = []
    for eid, entity in entities.items():
        idf_e = entity_idf.get(eid, 1.0)
        for chunk_id in entity.source_chunk_ids:
            chunk_idx = _parse_chunk_index(chunk_id)
            pos_decay = 1.0 / (1.0 + chunk_idx)
            w = idf_e * pos_decay
            belongs_to_edges.append(
                Edge(source=eid, target=chunk_id, type="belongs_to", weight=w)
            )
    all_edges.extend(belongs_to_edges)

    logger.info(
        "Graph built: %d entities, %d chunks, %d edges (merged %d entities)",
        len(entities),
        len(chunk_nodes),
        len(all_edges),
        len(merge_map),
    )

    return PivoGraph(entities=entities, chunks=chunk_nodes, edges=all_edges)
