"""图数据噪声注入：用于稳定性实验。

提供三类渐进式图结构噪声：
- adjacency:   随机添加 adjacency 边，模拟错误共现
- similarity:  随机添加 similarity 边，模拟错误语义相似
- merge:       随机合并实体节点，模拟错误实体合并
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Literal

from pivotrag.graph_builder import merge_entities
from pivotrag.models import Edge, PivoGraph

logger = logging.getLogger(__name__)

NoiseType = Literal["adjacency", "similarity", "merge"]


def _entity_entity_pair_set(graph: PivoGraph) -> set[tuple[str, str]]:
    """返回所有无向实体-实体边对（adjacency + similarity），用于查重。"""
    pairs: set[tuple[str, str]] = set()
    for e in graph.entity_subgraph_edges():
        pairs.add(tuple(sorted((e.source, e.target))))
    return pairs


def _min_positive_weight(graph: PivoGraph, edge_type: str) -> float:
    """获取某类型真实边中的最小正权重。"""
    weights = [e.weight for e in graph.edges if e.type == edge_type and e.weight > 0]
    return min(weights) if weights else 1.0


def _shared_document(graph: PivoGraph, u: str, v: str) -> bool:
    """判断两个实体是否共享同一个 source chunk / document。"""
    u_entity = graph.entities.get(u)
    v_entity = graph.entities.get(v)
    if u_entity is None or v_entity is None:
        return False
    u_docs = set(u_entity.source_chunk_ids)
    v_docs = set(v_entity.source_chunk_ids)
    return bool(u_docs & v_docs)


def inject_adjacency_noise(
    graph: PivoGraph,
    noise_ratio: float,
    seed: int = 42,
) -> PivoGraph:
    """随机添加 adjacency 假边，模拟错误共现。"""
    rng = random.Random(seed)
    noisy_graph = copy.deepcopy(graph)

    entity_ids = list(noisy_graph.entities.keys())
    if len(entity_ids) < 2:
        return noisy_graph

    existing_pairs = _entity_entity_pair_set(noisy_graph)
    n_real = len(existing_pairs)
    n_noise = int(round(noise_ratio * n_real))
    if n_noise <= 0:
        return noisy_graph

    fake_weight = _min_positive_weight(noisy_graph, "adjacency")
    added: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = n_noise * 200

    while len(added) < n_noise and attempts < max_attempts:
        attempts += 1
        u, v = rng.sample(entity_ids, 2)
        key = tuple(sorted((u, v)))
        if key in existing_pairs or key in added:
            continue
        # 避免同一 chunk/document 内的实体，防止假边“偶然正确”
        if _shared_document(noisy_graph, u, v):
            continue
        noisy_graph.edges.append(
            Edge(source=u, target=v, type="adjacency", weight=fake_weight)
        )
        added.add(key)

    logger.info(
        "Adjacency noise: added %d/%d fake edges (ratio=%.2f)",
        len(added),
        n_noise,
        noise_ratio,
    )
    return noisy_graph


def inject_similarity_noise(
    graph: PivoGraph,
    noise_ratio: float,
    seed: int = 42,
) -> PivoGraph:
    """随机添加 similarity 假边，模拟错误语义相似。"""
    rng = random.Random(seed)
    noisy_graph = copy.deepcopy(graph)

    entity_ids = list(noisy_graph.entities.keys())
    if len(entity_ids) < 2:
        return noisy_graph

    existing_pairs = _entity_entity_pair_set(noisy_graph)
    n_real_sim = sum(1 for e in noisy_graph.edges if e.type == "similarity")
    n_noise = int(round(noise_ratio * max(n_real_sim, 1)))
    if n_noise <= 0:
        return noisy_graph

    fake_weight = _min_positive_weight(noisy_graph, "similarity")
    added: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = n_noise * 200

    while len(added) < n_noise and attempts < max_attempts:
        attempts += 1
        u, v = rng.sample(entity_ids, 2)
        key = tuple(sorted((u, v)))
        if key in existing_pairs or key in added:
            continue
        if _shared_document(noisy_graph, u, v):
            continue
        noisy_graph.edges.append(
            Edge(source=u, target=v, type="similarity", weight=fake_weight)
        )
        added.add(key)

    logger.info(
        "Similarity noise: added %d/%d fake edges (ratio=%.2f)",
        len(added),
        n_noise,
        noise_ratio,
    )
    return noisy_graph


def inject_merge_noise(
    graph: PivoGraph,
    noise_ratio: float,
    seed: int = 42,
) -> PivoGraph:
    """随机合并实体节点，模拟错误实体合并。"""
    rng = random.Random(seed)
    noisy_graph = copy.deepcopy(graph)

    entity_ids = list(noisy_graph.entities.keys())
    if len(entity_ids) < 2:
        return noisy_graph

    n_entities = len(entity_ids)
    n_merges = int(round(noise_ratio * n_entities))
    if n_merges <= 0:
        return noisy_graph

    # 随机选择要合并的实体对，要求不共享文档
    merge_map: dict[str, str] = {}
    available = set(entity_ids)
    attempts = 0
    max_attempts = n_merges * 200

    while len(merge_map) < n_merges and attempts < max_attempts and len(available) >= 2:
        attempts += 1
        u, v = rng.sample(list(available), 2)
        if _shared_document(noisy_graph, u, v):
            continue
        # u -> v，保留 v
        merge_map[u] = v
        available.discard(u)

    if not merge_map:
        return noisy_graph

    new_entities, new_edges = merge_entities(
        noisy_graph.entities, noisy_graph.edges, merge_map
    )
    noisy_graph.entities = new_entities
    noisy_graph.edges = new_edges

    logger.info(
        "Merge noise: merged %d entities (ratio=%.2f)",
        len(merge_map),
        noise_ratio,
    )
    return noisy_graph


def inject_noise(
    graph: PivoGraph,
    noise_type: NoiseType,
    noise_ratio: float,
    seed: int = 42,
) -> PivoGraph:
    """统一入口：根据噪声类型调用对应注入函数。"""
    if noise_type == "adjacency":
        return inject_adjacency_noise(graph, noise_ratio, seed)
    if noise_type == "similarity":
        return inject_similarity_noise(graph, noise_ratio, seed)
    if noise_type == "merge":
        return inject_merge_noise(graph, noise_ratio, seed)
    raise ValueError(f"Unknown noise_type: {noise_type}")
