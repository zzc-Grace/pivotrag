"""Dual-path retrieval: PPR graph search + FAISS vector search."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import faiss
import networkx as nx
import numpy as np
import scipy.sparse as sp

from pivotrag.config import (
    BETA,
    DECOMPOSE_MAX_SUB_QUERIES,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    ENTITY_MATCH_TOP_N,
    K1_GRAPH,
    K2_VECTOR,
    LAMBDA_STR,
    LLM_MODEL,
    PPR_ALPHA,
    PPR_LOCAL_HOP,
    PPR_MAX_ITER,
    PPR_MODE,
    PPR_TOLERANCE,
    QUERY_TOP_N_ENTITIES,
)
from pivotrag.http_client import chat_completion, create_embeddings
from pivotrag.models import PivoGraph, RetrievalCandidate, RetrievalResult

logger = logging.getLogger(__name__)

_QUERY_EXTRACT_PROMPT = """\
You are a Named Entity Recognition (NER) system.
Extract all key entities (names, concepts, terms, locations, dates, organizations) explicitly mentioned in the following question.

Rules:
- Extract each unique entity only once.
- The entity name must exactly match the surface form in the original text.
- Do NOT include generic nouns, pronouns, or question words (who, what, where, when, how, etc.).

Return ONLY a JSON array of strings. Example: ["entity1", "entity2"]
If no clear entities, return [].
"""

_DECOMPOSE_PROMPT = """\
You are a question decomposition system.
Given a complex multi-hop question, break it down into simpler sub-questions that can be independently answered from separate knowledge sources.

Rules:
- Each sub-questions should be self-contained and answerable from a single document
- Sub-questions should cover ALL information needed to answer the original question
- Preserve the original entities and relationships
- Output ONLY a JSON array of strings (no explanations)

Examples:

Q: "What military overran much of Erich Zakowski's place of birth?"
A: ["Where was Erich Zakowski born?", "What military forces overran East Prussia?"]

Q: "Who is the mother of the person who plays Nim in Return to Nim's Island?"
A: ["Who plays Nim in Return to Nim's Island?", "Who is the mother of Bindi Irwin?"]

Q: "What city is in the county where the Red Cross is located?"
A: ["Where is the Red Cross located?", "What cities are in Stanly County, North Carolina?"]

Q: "When did the first mosque in the place where the Marshall Islands International Airport is located open?"
A: ["Where is the Marshall Islands International Airport located?", "When did the first mosque in the Marshall Islands open?"]

Q: "The German priest, who wanted to reform the religious denomination now the largest in the US, preached a sermon on Marian devotion soon before his death in which German state?"
A: ["What is the largest religious denomination in the US?", "Who was the German priest who started the Reformation?", "In which German state did Martin Luther die?"]
"""


class PivoRetriever:
    """检索器, 持有图和索引引用, 可复用多次查询."""

    def __init__(self, graph: PivoGraph) -> None:
        self._graph = graph
        self._entity_ids: list[str] = []
        self._chunk_ids: list[str] = []
        self._entity_index: faiss.IndexFlatIP | None = None
        self._chunk_index: faiss.IndexFlatIP | None = None
        self._entity_subgraph: nx.Graph = nx.Graph()
        self._entity_to_chunks_map: dict[str, list[str]] = {}
        self._entity_chunk_weights: dict[str, dict[str, float]] = {}  # eid -> {cid: weight}
        self._similarity_neighbors: dict[str, list[tuple[str, float]]] = {}
        # Pre-built sparse PPR structures (populated in _build_indexes)
        self._ppr_nodes: list[str] = []
        self._ppr_node_idx: dict[str, int] = {}
        self._W_sparse: sp.csr_matrix | None = None
        self._build_indexes()

    # ── Index pre-building ──

    def _build_indexes(self) -> None:
        """预构建 FAISS 索引 + entity 子图 + 反向索引 + 相似邻居索引."""
        # Entity FAISS index
        valid_entities = [
            (eid, self._graph.entities[eid])
            for eid in self._graph.entities
            if self._graph.entities[eid].embedding is not None
        ]
        if valid_entities:
            self._entity_ids = [eid for eid, _ in valid_entities]
            vecs = np.stack(
                [ent.embedding for _, ent in valid_entities]  # type: ignore[arg-type]
            ).astype(np.float32)
            faiss.normalize_L2(vecs)
            self._entity_index = faiss.IndexFlatIP(vecs.shape[1])
            self._entity_index.add(vecs)

        # Chunk FAISS index
        valid_chunks = [
            (cid, self._graph.chunks[cid])
            for cid in self._graph.chunks
            if self._graph.chunks[cid].embedding is not None
        ]
        if valid_chunks:
            self._chunk_ids = [cid for cid, _ in valid_chunks]
            vecs = np.stack(
                [ch.embedding for _, ch in valid_chunks]  # type: ignore[arg-type]
            ).astype(np.float32)
            faiss.normalize_L2(vecs)
            self._chunk_index = faiss.IndexFlatIP(vecs.shape[1])
            self._chunk_index.add(vecs)

        # Entity-only subgraph for PPR + similarity neighbor index
        for eid in self._graph.entities:
            self._entity_subgraph.add_node(eid)
            self._similarity_neighbors.setdefault(eid, [])

        for edge in self._graph.entity_subgraph_edges():
            self._entity_subgraph.add_edge(
                edge.source, edge.target, weight=edge.weight
            )
            # Build bidirectional similarity neighbor index
            if edge.type == "similarity":
                self._similarity_neighbors.setdefault(edge.source, []).append(
                    (edge.target, edge.weight)
                )
                self._similarity_neighbors.setdefault(edge.target, []).append(
                    (edge.source, edge.weight)
                )

        # Entity → chunks reverse index (with edge weights)
        for edge in self._graph.edges:
            if edge.type == "belongs_to":
                self._entity_to_chunks_map.setdefault(edge.source, []).append(
                    edge.target
                )
                self._entity_chunk_weights.setdefault(edge.source, {})[
                    edge.target
                ] = edge.weight

        # Pre-build sparse transition matrix for degree_normalized PPR
        self._build_sparse_transition_matrix()

    def _build_sparse_transition_matrix(self) -> None:
        """预构建 IDF 加权的行随机稀疏转移矩阵 (CSR 格式)."""
        self._ppr_nodes = list(self._entity_subgraph.nodes())
        self._ppr_node_idx = {node: i for i, node in enumerate(self._ppr_nodes)}
        n = len(self._ppr_nodes)

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for u in self._ppr_nodes:
            u_idx = self._ppr_node_idx[u]
            for v in self._entity_subgraph.neighbors(u):
                v_idx = self._ppr_node_idx[v]
                idf_v = self._idf(v)
                edge_data = self._entity_subgraph.get_edge_data(u, v)
                ew = edge_data.get("weight", 1.0) if edge_data else 1.0
                rows.append(u_idx)
                cols.append(v_idx)
                data.append(idf_v * ew)

        W = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        self._W_sparse = sp.diags(1.0 / row_sums) @ W
        logger.info("Sparse PPR matrix built: %d nodes, %d nonzeros", n, W.nnz)

    # ── Public API ──

    async def retrieve(
        self,
        query: str,
        llm_base_url: str,
        llm_api_key: str,
        emb_base_url: str,
        emb_api_key: str,
        k1: int = K1_GRAPH,
        k2: int = K2_VECTOR,
    ) -> RetrievalResult:
        """执行双路径检索.

        1. 从 query 抽取实体 → FAISS 定位种子实体 (n:n 匹配)
        2. PPR → top-k1 实体 → SIMILAR_TO 扩展 → 关联 chunks
        3. 向量检索 → top-k2 chunks
        """
        t_total = time.perf_counter()

        # Query embedding for vector path
        t0 = time.perf_counter()
        query_emb = await self._embed_query(query, emb_base_url, emb_api_key)
        t1 = time.perf_counter()
        logger.debug("[Time] Query embedding: %.3f s", t1 - t0)

        # Graph path: extract → locate → PPR → expand
        t0 = time.perf_counter()
        query_entities = await self._extract_query_entities(query, llm_base_url, llm_api_key)
        t1 = time.perf_counter()
        logger.debug("[Time] Query entity extraction: %.3f s", t1 - t0)

        t0 = time.perf_counter()
        seed_ids = await self._locate_seed_entities(query_entities, emb_base_url, emb_api_key)
        t1 = time.perf_counter()
        logger.debug("[Time] Seed entity location: %.3f s", t1 - t0)

        t0 = time.perf_counter()
        graph_candidates, ppr_entity_ids = self._ppr_search(
            seed_ids, k1, query_emb=query_emb
        )
        t1 = time.perf_counter()
        logger.debug("[Time] PPR search (%s): %.3f s", PPR_MODE, t1 - t0)

        # SIMILAR_TO entity expansion (reuses PPR result, no recompute)
        t0 = time.perf_counter()
        expanded_candidates = self._expand_entities_via_similarity(
            ppr_entity_ids, graph_candidates
        )
        if expanded_candidates:
            graph_candidates = expanded_candidates
        t1 = time.perf_counter()
        logger.debug("[Time] SIMILAR_TO expansion: %.3f s", t1 - t0)

        # Hybrid scoring: fuse structural + dense scores (方案4)
        t0 = time.perf_counter()
        graph_chunk_ids = [c.chunk_id for c in graph_candidates]
        dense_scores = self._dense_scores_for_chunks(query_emb, graph_chunk_ids)
        graph_candidates = self._hybrid_score_chunks(
            graph_candidates, dense_scores
        )
        t1 = time.perf_counter()
        logger.debug("[Time] Hybrid scoring: %.3f s", t1 - t0)

        # Vector path
        t0 = time.perf_counter()
        vector_candidates = self._vector_search(query_emb, k2)
        t1 = time.perf_counter()
        logger.debug("[Time] Vector search: %.3f s", t1 - t0)

        t_total_end = time.perf_counter()
        logger.info(
            "[Time] Total retrieve: %.3f s (graph=%d, vector=%d)",
            t_total_end - t_total,
            len(graph_candidates),
            len(vector_candidates),
        )

        return RetrievalResult(
            graph_candidates=graph_candidates,
            vector_candidates=vector_candidates,
            query_embedding=query_emb,
        )

    async def retrieve_with_decomposition(
        self,
        query: str,
        llm_base_url: str,
        llm_api_key: str,
        emb_base_url: str,
        emb_api_key: str,
        k1: int = K1_GRAPH,
        k2: int = K2_VECTOR,
    ) -> RetrievalResult:
        """查询分解 + 并行检索合并.

        1. LLM 将复杂问题分解为子问题
        2. 对每个子问题并行执行双路检索
        3. 合并去重所有候选结果
        """
        t_total = time.perf_counter()

        # 1. Decompose
        sub_queries = await self._decompose_query(query, llm_base_url, llm_api_key)
        if len(sub_queries) <= 1:
            # 单一问题，直接用原始检索
            return await self.retrieve(
                query, llm_base_url, llm_api_key, emb_base_url, emb_api_key, k1, k2,
            )

        # 2. Parallel retrieval for each sub-question
        async def _retrieve_one(sub_q: str) -> RetrievalResult:
            return await self.retrieve(
                sub_q, llm_base_url, llm_api_key, emb_base_url, emb_api_key, k1, k2,
            )

        sub_results = await asyncio.gather(*[_retrieve_one(sq) for sq in sub_queries])

        # 3. Merge candidates: deduplicate by chunk_id, keep highest score
        graph_best: dict[str, float] = {}
        vector_best: dict[str, float] = {}

        for sr in sub_results:
            for c in sr.graph_candidates:
                if c.chunk_id not in graph_best or c.score > graph_best[c.chunk_id]:
                    graph_best[c.chunk_id] = c.score
            for c in sr.vector_candidates:
                if c.chunk_id not in vector_best or c.score > vector_best[c.chunk_id]:
                    vector_best[c.chunk_id] = c.score

        merged_graph = [
            RetrievalCandidate(chunk_id=cid, score=score, source="graph")
            for cid, score in sorted(graph_best.items(), key=lambda x: x[1], reverse=True)
        ]
        merged_vector = [
            RetrievalCandidate(chunk_id=cid, score=score, source="vector")
            for cid, score in sorted(vector_best.items(), key=lambda x: x[1], reverse=True)
        ]

        t_total_end = time.perf_counter()
        logger.info(
            "[Time] Decomposed retrieve: %.3f s (%d sub-queries → %d graph + %d vector candidates)",
            t_total_end - t_total, len(sub_queries), len(merged_graph), len(merged_vector),
        )

        return RetrievalResult(
            graph_candidates=merged_graph,
            vector_candidates=merged_vector,
        )

    # ── Private: query decomposition ──

    async def _decompose_query(
        self, query: str, base_url: str, api_key: str, max_sub: int = DECOMPOSE_MAX_SUB_QUERIES,
    ) -> list[str]:
        """LLM 将复杂多跳问题分解为自包含的子问题列表."""
        try:
            content, _ = await chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _DECOMPOSE_PROMPT},
                    {"role": "user", "content": f'Q: "{query}"\nA:'},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                items = json.loads(match.group(0))
                sub_qs = [q for q in items if isinstance(q, str) and q.strip()][:max_sub]
                if sub_qs:
                    logger.info(
                        "Decomposed '%s' into %d sub-questions: %s",
                        query[:60], len(sub_qs), sub_qs,
                    )
                    return sub_qs
        except Exception:
            logger.warning("Query decomposition failed", exc_info=True)

        # Fallback: use original query as single sub-question
        return [query]

    # ── Private: query entity extraction ──

    async def _extract_query_entities(
        self, query: str, base_url: str, api_key: str
    ) -> list[str]:
        """LLM 从 query 中抽取实体名称列表."""
        try:
            content, _ = await chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _QUERY_EXTRACT_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                items = json.loads(match.group(0))
                return [e for e in items if isinstance(e, str)][:QUERY_TOP_N_ENTITIES]
        except Exception:
            logger.warning("Query entity extraction failed", exc_info=True)
        return []

    # ── Private: embedding ──

    async def _embed_query(
        self, query: str, base_url: str, api_key: str
    ) -> np.ndarray:
        """计算 query 的 embedding."""
        embeddings, _ = await create_embeddings(
            base_url=base_url,
            api_key=api_key,
            model=EMBEDDING_MODEL,
            input_texts=[query],
            dimensions=EMBEDDING_DIM,
        )
        return np.array(embeddings[0], dtype=np.float32)

    async def _locate_seed_entities(
        self,
        query_entity_names: list[str],
        base_url: str,
        api_key: str,
    ) -> list[str]:
        """N:N 实体匹配: 每个 query 实体独立 embedding 后在 entity FAISS 中搜索.

        每个 query 实体映射到 ENTITY_MATCH_TOP_N 个最相关的图中实体节点,
        所有匹配结果合并去重后作为 PPR 的种子实体。
        """
        if not query_entity_names or self._entity_index is None:
            return []

        embeddings, _ = await create_embeddings(
            base_url=base_url,
            api_key=api_key,
            model=EMBEDDING_MODEL,
            input_texts=query_entity_names,
            dimensions=EMBEDDING_DIM,
        )
        query_embs = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(query_embs)

        seed_ids: list[str] = []
        for i in range(len(query_embs)):
            scores, indices = self._entity_index.search(
                query_embs[i].reshape(1, -1), ENTITY_MATCH_TOP_N
            )
            matched_for_entity: list[str] = []
            for j in range(len(indices[0])):
                idx = int(indices[0][j])
                if 0 <= idx < len(self._entity_ids):
                    matched_for_entity.append(self._entity_ids[idx])
            if matched_for_entity:
                logger.debug(
                    "Query entity '%s' matched %d graph entities",
                    query_entity_names[i],
                    len(matched_for_entity),
                )
            seed_ids.extend(matched_for_entity)

        # Deduplicate preserving order
        return list(dict.fromkeys(seed_ids))

    # ── Private: PPR search (sync) ──

    def _get_ppr_top_entities(
        self, seed_entity_ids: list[str], k1: int
    ) -> list[str]:
        """获取 PPR 排名前 k1 的实体 ID 列表 (供 SIMILAR_TO 扩展使用)."""
        if not seed_entity_ids:
            return []

        valid_seeds = [
            sid for sid in seed_entity_ids if sid in self._entity_subgraph
        ]
        if not valid_seeds:
            return []

        if PPR_MODE == "degree_normalized":
            pr = self._degree_normalized_ppr(valid_seeds)
        else:
            pr = self._original_ppr(valid_seeds)

        if not pr:
            return []

        sorted_entities = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k1]
        return [eid for eid, _ in sorted_entities]

    def _ppr_search(
        self,
        seed_entity_ids: list[str],
        k1: int,
        query_emb: np.ndarray | None = None,
    ) -> tuple[list[RetrievalCandidate], list[str]]:
        """Personalized PageRank on entity subgraph.

        Uses query-adaptive local PPR when query_emb is provided (方案3).
        Chunk scoring uses structural sum formula (方案4).

        Returns (chunk_candidates, top_entity_ids) so callers don't need
        to recompute PPR just to get the entity ranking.
        """
        if not seed_entity_ids:
            return [], []

        valid_seeds = [
            sid for sid in seed_entity_ids if sid in self._entity_subgraph
        ]
        if not valid_seeds:
            return [], []

        # Query-adaptive local PPR (方案3)
        if query_emb is not None:
            pr = self._query_adaptive_ppr(valid_seeds, query_emb)
        elif PPR_MODE == "degree_normalized":
            pr = self._degree_normalized_ppr(valid_seeds)
        else:
            pr = self._original_ppr(valid_seeds)

        if not pr:
            return [], []

        # Top-k1 entities by PageRank value
        sorted_entities = sorted(pr.items(), key=lambda x: x[1], reverse=True)[
            :k1
        ]
        top_entity_ids = [eid for eid, _ in sorted_entities]

        # Structural chunk scoring: S_str(c|q) = sum_{e in N(c)} pi_q(e) * r(e,c)
        chunk_scores: dict[str, float] = {}
        for eid, pr_val in sorted_entities:
            for cid in self._entity_to_chunks_map.get(eid, []):
                ew = self._entity_chunk_weights.get(eid, {}).get(cid, 1.0)
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + pr_val * ew

        candidates = [
            RetrievalCandidate(chunk_id=cid, score=score, source="graph")
            for cid, score in sorted(
                chunk_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return candidates, top_entity_ids

    def _expand_entities_via_similarity(
        self,
        ppr_entity_ids: list[str],
        existing_candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """通过 SIMILAR_TO 关系扩展实体节点, 将扩展实体的关联 chunk 加入候选集.

        类似 pivotrag_old 的 expand_entities_with_similar_to 逻辑:
        1. 取 PPR top-k 实体
        2. 通过 similarity 边找到语义相近的其他实体
        3. 把这些扩展实体的关联 chunk 也加入候选集
        """
        if not ppr_entity_ids:
            return existing_candidates

        existing_chunk_ids = {c.chunk_id for c in existing_candidates}
        existing_entity_set = set(ppr_entity_ids)

        # Collect expanded entities from SIMILAR_TO neighbors
        expanded_entity_scores: dict[str, float] = {}
        for eid in ppr_entity_ids:
            neighbors = self._similarity_neighbors.get(eid, [])
            for neighbor_id, sim_weight in neighbors:
                if neighbor_id not in existing_entity_set:
                    # Use similarity weight as score, keep the max
                    if neighbor_id not in expanded_entity_scores or sim_weight > expanded_entity_scores[neighbor_id]:
                        expanded_entity_scores[neighbor_id] = sim_weight

        if not expanded_entity_scores:
            return existing_candidates

        # Map expanded entities to chunks
        new_chunk_scores: dict[str, float] = {}
        for eid, score in sorted(expanded_entity_scores.items(), key=lambda x: x[1], reverse=True):
            for cid in self._entity_to_chunks_map.get(eid, []):
                if cid not in existing_chunk_ids:
                    if cid not in new_chunk_scores or score > new_chunk_scores[cid]:
                        new_chunk_scores[cid] = score

        if not new_chunk_scores:
            return existing_candidates

        logger.info(
            "SIMILAR_TO expansion: %d PPR entities -> %d expanded entities -> %d new chunks",
            len(ppr_entity_ids),
            len(expanded_entity_scores),
            len(new_chunk_scores),
        )

        # Append expanded candidates after existing ones
        expanded_candidates = list(existing_candidates)
        for cid, score in sorted(new_chunk_scores.items(), key=lambda x: x[1], reverse=True):
            expanded_candidates.append(
                RetrievalCandidate(chunk_id=cid, score=score, source="graph")
            )

        return expanded_candidates

    def _original_ppr(self, valid_seeds: list[str]) -> dict[str, float]:
        """Standard Personalized PageRank via networkx."""
        n = len(valid_seeds)
        personalization = {sid: 1.0 / n for sid in valid_seeds}

        try:
            return nx.pagerank(
                self._entity_subgraph,
                alpha=PPR_ALPHA,
                personalization=personalization,
                weight="weight",
            )
        except Exception:
            logger.warning("Original PPR computation failed", exc_info=True)
            return {}

    def _idf(self, entity_id: str) -> float:
        """计算实体的 IDF 值: log(1 + |V_c| / (df(v) + 1))."""
        df_v = len(self._entity_to_chunks_map.get(entity_id, []))
        num_chunks = len(self._graph.chunks)
        return np.log(1.0 + num_chunks / (df_v + 1.0))

    def _degree_normalized_ppr(
        self, valid_seeds: list[str]
    ) -> dict[str, float]:
        """PivotRAG degree-normalized PPR with IDF-style hub suppression.

        Uses pre-built sparse transition matrix for O(nnz) per iteration
        instead of O(n^2) dense matrix-vector multiply.
        """
        if self._W_sparse is None or not self._ppr_nodes:
            return {}

        n = len(self._ppr_nodes)

        # IDF-weighted personalization vector
        p0 = np.zeros(n, dtype=np.float64)
        for sid in valid_seeds:
            idx = self._ppr_node_idx.get(sid)
            if idx is not None:
                p0[idx] = self._idf(sid)

        p0_sum = p0.sum()
        if p0_sum > 0:
            p0 /= p0_sum
        else:
            for sid in valid_seeds:
                idx = self._ppr_node_idx.get(sid)
                if idx is not None:
                    p0[idx] = 1.0 / len(valid_seeds)

        # PPR power iteration (sparse matrix-vector multiply)
        teleport_prob = 1.0 - PPR_ALPHA
        pi = p0.copy()
        WT = self._W_sparse.T.tocsr()

        for _ in range(PPR_MAX_ITER):
            pi_new = teleport_prob * p0 + PPR_ALPHA * (WT @ pi)
            if np.linalg.norm(pi_new - pi, ord=1) < PPR_TOLERANCE:
                break
            pi = pi_new

        return {self._ppr_nodes[i]: float(pi[i]) for i in range(n)}

    def _query_adaptive_ppr(
        self, valid_seeds: list[str], query_emb: np.ndarray
    ) -> dict[str, float]:
        """Query-Adaptive Pivot Navigation (方案3).

        1. Extract 2-hop local subgraph around seeds
        2. Compute query gate g(q, e_j) = 1 + beta * max(0, cos(q, e_j))
        3. Run PPR on gated local transition matrix
        """
        # Build local subgraph (union of ego graphs around each seed)
        local_nodes: set[str] = set()
        for sid in valid_seeds:
            if sid in self._entity_subgraph:
                try:
                    ego = nx.ego_graph(
                        self._entity_subgraph, sid, radius=PPR_LOCAL_HOP
                    )
                    local_nodes.update(ego.nodes())
                except Exception:
                    local_nodes.add(sid)

        if len(local_nodes) < 5:
            # Subgraph too small, fall back to full PPR
            return self._degree_normalized_ppr(valid_seeds)

        local_nodes_list = sorted(local_nodes)
        node_idx = {node: i for i, node in enumerate(local_nodes_list)}
        n = len(local_nodes_list)

        # Compute query gate for each node in subgraph
        query_emb_2d = query_emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_emb_2d)
        query_vec = query_emb_2d[0]

        gates: dict[str, float] = {}
        for node in local_nodes_list:
            ent = self._graph.entities.get(node)
            if ent is not None and ent.embedding is not None:
                ent_vec = ent.embedding.astype(np.float32)
                faiss.normalize_L2(ent_vec.reshape(1, -1))
                cos_sim = float(np.dot(query_vec, ent_vec))
                gates[node] = 1.0 + BETA * max(0.0, cos_sim)
            else:
                gates[node] = 1.0

        # Build gated local sparse transition matrix
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        subgraph = self._entity_subgraph.subgraph(local_nodes_list)
        for u in subgraph.nodes():
            u_idx = node_idx[u]
            for v in subgraph.neighbors(u):
                if v not in node_idx:
                    continue
                v_idx = node_idx[v]
                idf_v = self._idf(v)
                edge_data = subgraph.get_edge_data(u, v)
                ew = edge_data.get("weight", 1.0) if edge_data else 1.0
                rows.append(u_idx)
                cols.append(v_idx)
                data.append(idf_v * ew * gates[v])

        W = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        W_norm = sp.diags(1.0 / row_sums) @ W

        # IDF-weighted personalization vector
        p0 = np.zeros(n, dtype=np.float64)
        for sid in valid_seeds:
            idx = node_idx.get(sid)
            if idx is not None:
                p0[idx] = self._idf(sid)
        p0_sum = p0.sum()
        if p0_sum > 0:
            p0 /= p0_sum
        else:
            for sid in valid_seeds:
                idx = node_idx.get(sid)
                if idx is not None:
                    p0[idx] = 1.0 / len(valid_seeds)

        # Power iteration
        teleport_prob = 1.0 - PPR_ALPHA
        pi = p0.copy()
        WT = W_norm.T.tocsr()

        for _ in range(PPR_MAX_ITER):
            pi_new = teleport_prob * p0 + PPR_ALPHA * (WT @ pi)
            if np.linalg.norm(pi_new - pi, ord=1) < PPR_TOLERANCE:
                break
            pi = pi_new

        return {local_nodes_list[i]: float(pi[i]) for i in range(n)}

    # ── Private: vector search (sync) ──

    def _vector_search(
        self,
        query_embedding: np.ndarray,
        k2: int,
    ) -> list[RetrievalCandidate]:
        """FAISS 向量检索 on chunk embeddings."""
        if self._chunk_index is None:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        k = min(k2, self._chunk_index.ntotal)
        if k <= 0:
            return []

        scores, indices = self._chunk_index.search(query, k)

        return [
            RetrievalCandidate(
                chunk_id=self._chunk_ids[int(indices[0][i])],
                score=float(scores[0][i]),
                source="vector",
            )
            for i in range(len(indices[0]))
            if int(indices[0][i]) < len(self._chunk_ids)
        ]

    def _dense_scores_for_chunks(
        self,
        query_embedding: np.ndarray,
        chunk_ids: list[str],
    ) -> dict[str, float]:
        """Compute dense cosine similarity for specific chunk IDs via FAISS."""
        if self._chunk_index is None or not chunk_ids:
            return {}

        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Search enough neighbors to cover requested chunks
        n_search = min(self._chunk_index.ntotal, max(len(chunk_ids) * 3, 100))
        scores, indices = self._chunk_index.search(query, n_search)

        result: dict[str, float] = {}
        target_set = set(chunk_ids)
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx < len(self._chunk_ids):
                cid = self._chunk_ids[idx]
                if cid in target_set:
                    result[cid] = float(scores[0][i])
        return result

    @staticmethod
    def _hybrid_score_chunks(
        graph_candidates: list[RetrievalCandidate],
        dense_scores: dict[str, float],
        lambda_str: float = LAMBDA_STR,
    ) -> list[RetrievalCandidate]:
        """Fuse structural and dense scores: S = lambda * S_str + (1-lambda) * S_dense."""
        hybrid: list[RetrievalCandidate] = []
        for c in graph_candidates:
            s_str = c.score
            s_dense = dense_scores.get(c.chunk_id, 0.0)
            s_hybrid = lambda_str * s_str + (1.0 - lambda_str) * s_dense
            hybrid.append(
                RetrievalCandidate(
                    chunk_id=c.chunk_id, score=s_hybrid, source="graph"
                )
            )
        hybrid.sort(key=lambda x: x.score, reverse=True)
        return hybrid
