"""PivoRAG pipeline: index building, querying, and batch evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

import numpy as np

from pivotrag.chunker import chunk_documents, load_documents
from pivotrag.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    ENTITY_EXTRACTOR_BACKEND,
    K1_GRAPH,
    K2_VECTOR,
    K3_FINAL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    QUERY_DECOMPOSE,
)
from pivotrag.entity_extractor import batch_extract_entities as llm_batch_extract_entities
from pivotrag.spacy_entity_extractor import batch_extract_entities as spacy_batch_extract_entities
from pivotrag.eval import compute_metrics, compute_retrieval_recall
from pivotrag.graph_builder import _make_entity_id, build_graph
from pivotrag.graph_storage import graph_exists, load_graph, save_graph
from pivotrag.http_client import chat_completion, create_embeddings
from pivotrag.models import RankedChunk, TokenUsage
from pivotrag.reranker import PivoReranker
from pivotrag.retriever import PivoRetriever

logger = logging.getLogger(__name__)

# ── Prompt for answer generation ──

_ANSWER_SYSTEM_PROMPT = """\
You are a precise QA system. Your job is to extract the shortest correct answer from the retrieved context.

## Rules
1. Output ONLY the exact answer phrase — no full sentences, no extra words, no explanations.
2. Match the answer type to the question:
   - "Who" → a person or organization name
   - "What" → a thing, concept, or short phrase
   - "When" / "What year" → a date or time period
   - "Where" → a place name
   - "How many" / "How much" → a number
   - "Yes/No" → "yes" or "no"
3. For comparison questions ("which is farther/larger/older"), output only the chosen entity name.
4. For multi-hop questions, decompose the question into sub-questions, find the answer to each sub-question from the context, then combine them to get the final answer.
5. ALWAYS try your best to find an answer from the context. Look carefully across ALL provided context passages — the answer may be spread across multiple passages or require connecting information from different sources.
6. ONLY output "Cannot be answered based on the provided context." if you have thoroughly examined all context passages and absolutely cannot find any relevant information.

## Examples

### Single-hop
Q: "Who composed the Symphony No. 9?"
Context: [1] Symphony No. 9 in D minor, Op. 125 is a choral symphony composed by Ludwig van Beethoven.
A: Ludwig van Beethoven

Q: "What years did the company operate?"
Context: [1] The company operated throughout the 1970s and 1980s in South America.
A: 1970s and 1980s

### Comparison
Q: "Are Alice and Bob from the same city?"
Context: [1] Alice was born in Paris. [2] Bob was born in London.
A: no

Q: "Which team won the 2020 championship?"
Context: [1] The Lakers defeated the Heat in the 2020 NBA Finals.
A: Lakers

### Multi-hop (reason step by step)
Q: "What is the capital of the country where the inventor of the telephone was born?"
Context: [1] Alexander Graham Bell invented the telephone. He was born in Edinburgh, Scotland. [2] Edinburgh is the capital of Scotland.
Step 1: The inventor of the telephone is Alexander Graham Bell.
Step 2: He was born in Scotland.
Step 3: The capital of Scotland is Edinburgh.
A: Edinburgh

Q: "Who sings Home Alone Tonight with the singer of 'I Don't Want This Night to End'?"
Context: [1] "I Don't Want This Night to End" is a song by Luke Bryan. [2] "Home Alone Tonight" is a duet by Luke Bryan and Karen Fairchild.
Step 1: The singer of "I Don't Want This Night to End" is Luke Bryan.
Step 2: "Home Alone Tonight" is sung by Luke Bryan and Karen Fairchild.
A: Karen Fairchild

Q: "What military overran much of Erich Zakowski's place of birth?"
Context: [1] Erich Zakowski was born in Schneidemühl, Germany. [2] During WWI, Imperial Russian troops overran much of Schneidemühl.
Step 1: Erich Zakowski was born in Schneidemühl.
Step 2: Imperial Russian troops overran much of Schneidemühl.
A: Imperial Russian troops

---

## Your Task

Question:
{question}

Retrieved Context:
{context}

Think step by step if the question requires multi-hop reasoning. Then output ONLY the final answer:
"""


# ── Embedding helpers ──


async def _batch_embed(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_concurrency: int = 5,
    usage_counter: TokenUsage | None = None,
) -> list[np.ndarray]:
    """批量计算 embedding, 分批并发请求."""
    if not texts:
        return []

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    sem = asyncio.Semaphore(max_concurrency)
    # Pre-allocate result slots to preserve order
    results: list[list[np.ndarray]] = [None] * len(batches)  # type: ignore[list-item]

    async def _embed_batch(idx: int, batch: list[str]) -> None:
        async with sem:
            embeddings, usage = await create_embeddings(
                base_url=EMBEDDING_BASE_URL,
                api_key=EMBEDDING_API_KEY,
                model=model,
                input_texts=batch,
                dimensions=EMBEDDING_DIM,
            )
            if usage_counter and usage:
                usage_counter.embedding_input_tokens += usage.get("prompt_tokens", 0)
            results[idx] = [
                np.array(emb, dtype=np.float32) for emb in embeddings
            ]

    await asyncio.gather(*[_embed_batch(i, b) for i, b in enumerate(batches)])
    return [emb for batch_result in results for emb in batch_result]


# ── Phase 1: Index Building ──


async def build_index(root_path: str) -> TokenUsage:
    """构建 PivoRAG 索引 (完全独立, 不依赖 GraphRAG).

    流程:
    1. 从 input_path 加载原始文档并分块
    2. 异步批量抽取实体
    3. 计算实体 + chunk embedding
    4. 构建图
    5. 持久化

    Returns:
        TokenUsage 汇总
    """
    t_total_start = time.perf_counter()
    root = root_path.rstrip("/")
    output_dir = f"{root}/output/pivorag"
    usage = TokenUsage()

    if graph_exists(output_dir):
        logger.info("Index already exists at %s, skipping build", output_dir)
        return usage

    # 1. Load & chunk documents
    t0 = time.perf_counter()
    documents = load_documents(root)
    if not documents:
        logger.error("No documents found in %s", root)
        return usage
    logger.info("Loaded %d documents", len(documents))

    chunk_nodes_list = chunk_documents(documents)
    if not chunk_nodes_list:
        logger.error("No chunks produced")
        return usage

    chunk_nodes = {c.id: c for c in chunk_nodes_list}
    chunks_data: list[tuple[str, str]] = [(c.id, c.text) for c in chunk_nodes_list]
    t1 = time.perf_counter()
    logger.info("[Time] Load & chunk: %.3f s", t1 - t0)

    # 2. Extract entities
    t0 = time.perf_counter()
    logger.info("Extracting entities from %d chunks (backend=%s)...", len(chunks_data), ENTITY_EXTRACTOR_BACKEND)
    if ENTITY_EXTRACTOR_BACKEND == "spacy":
        chunk_entities = await spacy_batch_extract_entities(chunks_data)
    else:
        chunk_entities = await llm_batch_extract_entities(
            chunks_data, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, usage_counter=usage
        )

    total_entities = sum(len(v) for v in chunk_entities.values())
    t1 = time.perf_counter()
    logger.info(
        "Extracted %d entities from %d chunks (%.3f s)",
        total_entities,
        len(chunks_data),
        t1 - t0,
    )

    # 3. Compute entity + chunk embeddings in parallel
    t0 = time.perf_counter()

    # Prepare entity embedding inputs
    entity_texts: list[str] = []
    entity_eids: list[str] = []
    for chunk_id, ent_list in chunk_entities.items():
        for idx, (name, desc) in enumerate(ent_list):
            eid = _make_entity_id(chunk_id, idx)
            entity_texts.append(f"{name}: {desc}")
            entity_eids.append(eid)

    # Prepare chunk embedding inputs
    chunk_texts = [ctext for _, ctext in chunks_data]
    chunk_ids_ordered = [cid for cid, _ in chunks_data]

    logger.info(
        "Computing embeddings for %d entities + %d chunks (parallel)...",
        len(entity_texts), len(chunk_texts),
    )

    # Run both embedding tasks concurrently
    entity_embs_list, chunk_embs_list = await asyncio.gather(
        _batch_embed(entity_texts, usage_counter=usage),
        _batch_embed(chunk_texts, usage_counter=usage),
    )

    entity_embeddings = dict(zip(entity_eids, entity_embs_list))
    for cid, emb_arr in zip(chunk_ids_ordered, chunk_embs_list):
        chunk_nodes[cid].embedding = emb_arr

    t1 = time.perf_counter()
    logger.info("[Time] All embeddings (parallel): %.3f s", t1 - t0)

    # 4. Build graph
    t0 = time.perf_counter()
    logger.info("Building graph...")
    graph = build_graph(chunk_entities, chunk_nodes, entity_embeddings)
    t1 = time.perf_counter()
    logger.info("[Time] Build graph: %.3f s", t1 - t0)

    # 5. Save
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    save_graph(graph, output_dir)
    t1 = time.perf_counter()
    logger.info("[Time] Save graph: %.3f s", t1 - t0)

    t_total_end = time.perf_counter()
    logger.info(
        "Index built and saved to %s (total %.3f s)", output_dir, t_total_end - t_total_start
    )
    logger.info(
        "Token usage — LLM: %d in + %d out, Embedding: %d in",
        usage.llm_input_tokens, usage.llm_output_tokens, usage.embedding_input_tokens,
    )
    return usage


# ── Phase 2: Query ──


async def query(
    root_path: str,
    question: str,
) -> list[RankedChunk]:
    """单次查询: 双路检索 + 重排.

    Returns:
        Reranked top-k3 chunks.
    """
    root = root_path.rstrip("/")
    output_dir = f"{root}/output/pivorag"

    graph = load_graph(output_dir)

    t0 = time.perf_counter()
    retriever = PivoRetriever(graph)
    retrieve_fn = (
        retriever.retrieve_with_decomposition
        if QUERY_DECOMPOSE
        else retriever.retrieve
    )
    result = await retrieve_fn(
        question, LLM_BASE_URL, LLM_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_API_KEY,
        K1_GRAPH, K2_VECTOR,
    )

    reranker = PivoReranker()
    ranked = await reranker.rerank(question, result, graph, K3_FINAL)
    t1 = time.perf_counter()
    logger.info("[Time] Query retrieval + rerank: %.3f s", t1 - t0)

    return ranked


# ── Phase 3: Batch Evaluate ──


async def batch_evaluate(
    root_path: str,
    qa_file: str,
    output_file: str,
    limit: int | None = None,
) -> dict:
    """批量评估: 加载 qa-pairs → 检索 → 生成答案 → 计算 F1/EM + Recall@k.

    额外评估检索召回率（recall@5、recall@10），采用 document-level 不完全匹配：
    只要检索到的 chunk 所属文档标题与 supporting_facts 中的标题归一化后一致即算命中。

    Args:
        root_path: 根目录路径
        qa_file: QA对文件路径
        output_file: 输出结果文件路径
        limit: 限制评估的QA对数量，None表示评估所有

    Returns:
        {"avg_f1": float, "avg_em": float, "avg_recall@5": float,
         "avg_recall@10": float, "total": int}
    """
    root = root_path.rstrip("/")
    output_dir = f"{root}/output/pivorag"

    graph = load_graph(output_dir)
    retriever = PivoRetriever(graph)
    reranker = PivoReranker()

    # Load QA pairs
    with open(qa_file, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    valid_pairs = [
        p
        for p in qa_pairs
        if isinstance(p.get("question"), str) and isinstance(p.get("answer"), str)
    ]

    # 限制评估数量
    if limit is not None:
        valid_pairs = valid_pairs[:limit]

    logger.info("Evaluating %d QA pairs...", len(valid_pairs))

    predictions: list[dict] = []
    sem = asyncio.Semaphore(10)

    async def _process_pair(pair: dict) -> dict:
        async with sem:
            question = pair["question"]

            # 提取 gold supporting facts
            supporting_facts = pair.get("supporting_facts", [])
            gold_titles: set[str] = set()
            gold_facts_detail: list[dict] = []
            if isinstance(supporting_facts, list):
                for fact in supporting_facts:
                    if isinstance(fact, (list, tuple)) and len(fact) > 0:
                        title = str(fact[0])
                        gold_titles.add(title)
                        gold_facts_detail.append(
                            {"title": title, "paragraph_index": fact[1] if len(fact) > 1 else None}
                        )
                    elif isinstance(fact, str):
                        gold_titles.add(fact)
                        gold_facts_detail.append({"title": fact})

            t0 = time.perf_counter()
            try:
                retrieve_fn = (
                    retriever.retrieve_with_decomposition
                    if QUERY_DECOMPOSE
                    else retriever.retrieve
                )
                result = await retrieve_fn(
                    question, LLM_BASE_URL, LLM_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_API_KEY,
                )
                # 评估召回率需要至少 10 个结果
                ranked_all = await reranker.rerank(
                    question, result, graph, max(K3_FINAL, 10)
                )
                ranked = ranked_all[:K3_FINAL]

                # 提取前 10 个检索结果对应的文档标题（用于计算 recall）
                retrieved_titles: list[str] = []
                retrieved_docs_detail: list[dict] = []
                for c in ranked_all[:10]:
                    chunk = graph.chunks.get(c.chunk_id)
                    if chunk:
                        title = chunk.document_ids[0] if chunk.document_ids else ""
                        retrieved_titles.append(title)
                        retrieved_docs_detail.append(
                            {
                                "chunk_id": c.chunk_id,
                                "title": title,
                                "text": chunk.text[:300],
                            }
                        )

                recall_metrics = compute_retrieval_recall(
                    retrieved_titles, gold_titles, ks=[5, 10]
                )

                context = "\n".join(
                    f"[{i + 1}] {c.chunk_text}" for i, c in enumerate(ranked)
                )

                prompt = _ANSWER_SYSTEM_PROMPT.format(question=question, context=context)
                answer, _ = await chat_completion(
                    base_url=LLM_BASE_URL,
                    api_key=LLM_API_KEY,
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a precise QA system. For multi-hop questions, reason step by step. Always try to find an answer from the context before saying it cannot be answered. Output ONLY the final answer phrase with no extra words."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                t1 = time.perf_counter()
                logger.debug("[Time] Single query (retrieve+rerank+generate): %.3f s", t1 - t0)

                return {
                    "id": pair.get("id", ""),
                    "llm_answer": answer,
                    "ground_truth": pair["answer"],
                    "question": question,
                    "supporting_facts": gold_facts_detail,
                    "retrieved_docs": retrieved_docs_detail,
                    "recall@5": recall_metrics.get("recall@5", 0.0),
                    "recall@10": recall_metrics.get("recall@10", 0.0),
                }
            except Exception:
                t1 = time.perf_counter()
                logger.error(
                    "Failed to process question '%s' (%.3f s), skipping",
                    question[:80], t1 - t0, exc_info=True,
                )
                return {
                    "id": pair.get("id", ""),
                    "llm_answer": "",
                    "ground_truth": pair["answer"],
                    "question": question,
                    "supporting_facts": gold_facts_detail,
                    "retrieved_docs": [],
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "error": True,
                }

    t_eval_start = time.perf_counter()
    total_pairs = len(valid_pairs)

    async def _process_with_progress(idx: int, pair: dict) -> dict:
        result = await _process_pair(pair)
        if (idx + 1) % 10 == 0 or idx + 1 == total_pairs:
            logger.info("Progress: %d/%d completed", idx + 1, total_pairs)
        return result

    tasks = [_process_with_progress(i, p) for i, p in enumerate(valid_pairs)]
    results = await asyncio.gather(*tasks)
    predictions.extend(results)
    t_eval_end = time.perf_counter()

    error_count = sum(1 for p in predictions if p.get("error"))
    metrics = compute_metrics(predictions)
    metrics["eval_time_seconds"] = round(t_eval_end - t_eval_start, 3)
    metrics["error_count"] = error_count

    # 汇总平均召回率
    if predictions:
        metrics["avg_recall@5"] = sum(
            p.get("recall@5", 0.0) for p in predictions
        ) / len(predictions)
        metrics["avg_recall@10"] = sum(
            p.get("recall@10", 0.0) for p in predictions
        ) / len(predictions)
    else:
        metrics["avg_recall@5"] = 0.0
        metrics["avg_recall@10"] = 0.0

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "predictions": predictions},
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Evaluation complete: F1=%.4f, EM=%.4f, Recall@5=%.4f, Recall@10=%.4f, "
        "total=%d, errors=%d, total_time=%.3f s",
        metrics["avg_f1"],
        metrics["avg_em"],
        metrics["avg_recall@5"],
        metrics["avg_recall@10"],
        metrics["total"],
        error_count,
        t_eval_end - t_eval_start,
    )
    return metrics


# ── CLI ──


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m pivotrag.pipeline build <input_path>")
        print("  python -m pivotrag.pipeline query <root_path> <question>")
        print("  python -m pivotrag.pipeline eval <root_path> <qa_file> [output_file] [limit]")
        sys.exit(1)

    cmd = sys.argv[1]
    root = sys.argv[2]

    if cmd == "build":
        await build_index(root)
    elif cmd == "query":
        question = sys.argv[3] if len(sys.argv) > 3 else "What is PivoRAG?"
        results = await query(root, question)
        print(f"\nQuestion: {question}")
        print(f"Top {len(results)} chunks:\n")
        for i, r in enumerate(results):
            print(f"  [{i + 1}] score={r.rerank_score:.4f} | {r.chunk_text[:120]}...")
    elif cmd == "eval":
        qa_file = (
            sys.argv[3] if len(sys.argv) > 3 else f"{root}/qa-pairs/qa-pairs.json"
        )
        out_file = (
            sys.argv[4] if len(sys.argv) > 4 else f"{root}/output/pivorag_eval.json"
        )
        limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
        metrics = await batch_evaluate(root, qa_file, out_file, limit)
        print(
            f"\nF1={metrics['avg_f1']:.4f}, EM={metrics['avg_em']:.4f}, "
            f"Recall@5={metrics['avg_recall@5']:.4f}, Recall@10={metrics['avg_recall@10']:.4f}, "
            f"N={metrics['total']}, Errors={metrics.get('error_count', 0)}"
        )
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
