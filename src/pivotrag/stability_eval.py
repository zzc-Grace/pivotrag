"""PivoRAG 稳定性评估（单次运行）。

指定噪声类型和噪声率，对单个带噪图进行评估，输出 recall@5 / f1 / acc 指标。
不生成效果图，不循环多组噪声，便于外部调度器批量调用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from pivotrag.config import K3_FINAL
from pivotrag.eval import compute_metrics, compute_retrieval_recall, exact_match_score, f1_score
from pivotrag.graph_storage import load_graph
from pivotrag.http_client import chat_completion
from pivotrag.models import PivoGraph
from pivotrag.noise_injection import NoiseType, inject_noise
from pivotrag.pipeline import _ANSWER_SYSTEM_PROMPT
from pivotrag.reranker import PivoReranker
from pivotrag.retriever import PivoRetriever

logger = logging.getLogger(__name__)


def _chunk_title(graph: PivoGraph, chunk_id: str) -> str:
    chunk = graph.chunks.get(chunk_id)
    if chunk and chunk.document_ids:
        return chunk.document_ids[0]
    return ""


async def _evaluate_single(
    retriever: PivoRetriever,
    reranker: PivoReranker,
    graph: PivoGraph,
    pair: dict,
    idx: int,
    total: int,
) -> dict:
    """评估单个 QA pair。"""
    question = pair["question"]
    ground_truth = pair["answer"]

    supporting_facts = pair.get("supporting_facts", [])
    gold_titles: set[str] = set()
    for fact in supporting_facts:
        if isinstance(fact, (list, tuple)) and len(fact) > 0:
            gold_titles.add(str(fact[0]))
        elif isinstance(fact, str):
            gold_titles.add(fact)

    try:
        from pivotrag.config import (
            EMBEDDING_API_KEY,
            EMBEDDING_BASE_URL,
            LLM_API_KEY,
            LLM_BASE_URL,
        )

        result = await retriever.retrieve(
            question,
            LLM_BASE_URL,
            LLM_API_KEY,
            EMBEDDING_BASE_URL,
            EMBEDDING_API_KEY,
        )

        ranked_all = await reranker.rerank(
            question, result, graph, max(K3_FINAL, 10)
        )
        ranked = ranked_all[:K3_FINAL]

        retrieved_titles: list[str] = []
        for c in ranked_all[:10]:
            title = _chunk_title(graph, c.chunk_id)
            if title:
                retrieved_titles.append(title)

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
            model=os.getenv("PIVORAG_LLM_MODEL", "your-llm-model"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise QA system. For multi-hop questions, "
                        "reason step by step. Always try to find an answer from "
                        "the context before saying it cannot be answered. Output "
                        "ONLY the final answer phrase with no extra words."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )

        f1, _, _ = f1_score(answer, ground_truth)
        em = exact_match_score(answer, ground_truth)

        logger.info(
            "[%d/%d] recall@5=%.2f f1=%.2f em=%.2f | Q: %s",
            idx,
            total,
            recall_metrics.get("recall@5", 0.0),
            f1,
            em,
            question[:60],
        )

        return {
            "id": pair.get("id", ""),
            "question": question,
            "llm_answer": answer,
            "ground_truth": ground_truth,
            "recall@5": recall_metrics.get("recall@5", 0.0),
            "recall@10": recall_metrics.get("recall@10", 0.0),
            "f1": f1,
            "em": em,
        }
    except Exception:
        logger.error("Failed to evaluate question: %s", question[:80], exc_info=True)
        return {
            "id": pair.get("id", ""),
            "question": question,
            "llm_answer": "",
            "ground_truth": ground_truth,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "f1": 0.0,
            "em": 0.0,
            "error": True,
        }


async def evaluate_graph(
    graph: PivoGraph,
    qa_pairs: list[dict],
    limit: int | None = None,
) -> dict:
    """评估一个图在 QA 集上的指标。"""
    if limit is not None:
        qa_pairs = qa_pairs[:limit]

    logger.info("Building retriever indexes...")
    retriever = PivoRetriever(graph)
    reranker = PivoReranker()
    logger.info("Retriever ready. Starting evaluation on %d QA pairs.", len(qa_pairs))

    sem = asyncio.Semaphore(10)
    total = len(qa_pairs)

    async def _wrap(idx: int, pair: dict) -> dict:
        async with sem:
            return await _evaluate_single(retriever, reranker, graph, pair, idx, total)

    results = await asyncio.gather(*[_wrap(i + 1, p) for i, p in enumerate(qa_pairs)])

    metrics = compute_metrics(
        [
            {"llm_answer": r["llm_answer"], "ground_truth": r["ground_truth"]}
            for r in results
        ]
    )
    metrics["avg_recall@5"] = sum(r["recall@5"] for r in results) / len(results)
    metrics["avg_recall@10"] = sum(r["recall@10"] for r in results) / len(results)
    metrics["predictions"] = results
    return metrics


async def run_single_eval(
    clean_graph_dir: str,
    qa_file: str,
    noise_type: NoiseType,
    noise_ratio: float,
    output_file: str,
    seed: int = 42,
    limit: int | None = None,
) -> dict:
    """单次运行：注入指定噪声，评估并保存指标。"""
    logger.info("Loading clean graph from %s", clean_graph_dir)
    clean_graph = load_graph(clean_graph_dir)
    logger.info(
        "Clean graph loaded: %d entities, %d chunks, %d edges",
        len(clean_graph.entities),
        len(clean_graph.chunks),
        len(clean_graph.edges),
    )

    with open(qa_file, encoding="utf-8") as f:
        qa_pairs = json.load(f)
    qa_pairs = [
        p
        for p in qa_pairs
        if isinstance(p.get("question"), str) and isinstance(p.get("answer"), str)
    ]
    logger.info("Loaded %d valid QA pairs from %s", len(qa_pairs), qa_file)

    logger.info(
        "Injecting %s noise (ratio=%.2f, seed=%d)...",
        noise_type,
        noise_ratio,
        seed,
    )
    noisy_graph = inject_noise(clean_graph, noise_type, noise_ratio, seed=seed)
    logger.info(
        "Noisy graph: %d entities, %d chunks, %d edges",
        len(noisy_graph.entities),
        len(noisy_graph.chunks),
        len(noisy_graph.edges),
    )

    logger.info("Evaluating noisy graph on %d QA pairs...", len(qa_pairs))
    metrics = await evaluate_graph(noisy_graph, qa_pairs, limit=limit)

    summary = {
        "noise_type": noise_type,
        "noise_ratio": noise_ratio,
        "seed": seed,
        "n_qa_pairs": len(qa_pairs) if limit is None else min(limit, len(qa_pairs)),
        "metrics": {
            "recall@5": round(metrics["avg_recall@5"], 4),
            "recall@10": round(metrics["avg_recall@10"], 4),
            "f1": round(metrics["avg_f1"], 4),
            "em": round(metrics["avg_em"], 4),
        },
        "avg_f1": round(metrics["avg_f1"], 4),
        "avg_em": round(metrics["avg_em"], 4),
        "predictions": metrics["predictions"],
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "Results saved to %s: recall@5=%.4f, f1=%.4f, em=%.4f",
        output_path,
        summary["metrics"]["recall@5"],
        summary["metrics"]["f1"],
        summary["metrics"]["em"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PivoRAG single noise evaluation")
    parser.add_argument(
        "clean_graph_dir",
        help="Path to the clean graph directory (contains pivorag_nodes.parquet etc.)",
    )
    parser.add_argument("qa_file", help="Path to QA pairs JSON file")
    parser.add_argument(
        "--noise-type",
        choices=["adjacency", "similarity", "merge"],
        default="adjacency",
        help="Type of noise to inject (default: adjacency)",
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.0,
        help="Noise ratio (default: 0.0)",
    )
    parser.add_argument(
        "--output-file",
        default="output/stability/single_eval.json",
        help="Output JSON file path (default: output/stability/single_eval.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise injection (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of QA pairs for quick testing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(
        run_single_eval(
            clean_graph_dir=args.clean_graph_dir,
            qa_file=args.qa_file,
            noise_type=args.noise_type,
            noise_ratio=args.noise_ratio,
            output_file=args.output_file,
            seed=args.seed,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
