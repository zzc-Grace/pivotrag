"""Evaluation metrics: F1, Exact Match, and Retrieval Recall."""

from __future__ import annotations

import re
import string
from collections import Counter


def _normalize_answer(s: str) -> str:
    """Normalize answer for comparison (copied from util_v1)."""

    def _remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def _white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def _remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def _lower(text: str) -> str:
        return text.lower()

    return _white_space_fix(_remove_articles(_remove_punc(_lower(s))))


def _normalize_text(s: str) -> str:
    """Normalize text for fuzzy matching.

    Handles the parentheses-vs-underscore mismatch caused by filesystem-safe
    title sanitisation in the data-export step (build_dataset.py).
    """
    # Replace characters that filesystem sanitisation converts to underscores
    s = re.sub(r'[<>:"/\\|?*()]', "_", s)
    # Collapse consecutive underscores/spaces and strip leading/trailing
    s = re.sub(r"[_\s]+", " ", s).strip(" _")
    return s.lower()


def compute_retrieval_recall(
    retrieved_doc_titles: list[str],
    gold_doc_titles: set[str],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """计算 retrieval recall@k（不完全匹配，document-level）。

    对标题做归一化后比较（忽略大小写和多余空格），只要检索结果中的文档标题
    与 gold supporting_facts 中的某个标题归一化后一致即算命中。

    Args:
        retrieved_doc_titles: 按检索顺序排列的文档标题列表（可含重复）。
        gold_doc_titles: 标准事实段落文档标题集合。
        ks: 要计算的 k 值，默认 [5, 10]。

    Returns:
        {"recall@5": float, "recall@10": float, ...}
    """
    if ks is None:
        ks = [5, 10]

    norm_gold = {_normalize_text(t) for t in gold_doc_titles}
    if not norm_gold:
        return {f"recall@{k}": 0.0 for k in ks}

    # 对检索结果去重，保留顺序
    seen: set[str] = set()
    deduped: list[str] = []
    for t in retrieved_doc_titles:
        nt = _normalize_text(t)
        if nt not in seen:
            seen.add(nt)
            deduped.append(nt)

    results: dict[str, float] = {}
    for k in ks:
        top_k = set(deduped[:k])
        hits = len(norm_gold & top_k)
        results[f"recall@{k}"] = hits / len(norm_gold)
    return results


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute token-level F1, precision, recall."""
    norm_pred = _normalize_answer(prediction)
    norm_gt = _normalize_answer(ground_truth)

    zero = (0.0, 0.0, 0.0)

    if norm_pred in ("yes", "no", "insufficient information") and norm_pred != norm_gt:
        return zero
    if norm_gt in ("yes", "no") and norm_gt not in norm_pred:
        return zero

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """Compute exact match (0 or 1)."""
    return int(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def compute_metrics(
    predictions: list[dict],
) -> dict:
    """计算平均 F1 和 EM.

    Args:
        predictions: [{"id": ..., "answer": ..., "ground_truth": ...}]

    Returns:
        {"avg_f1": float, "avg_em": float, "total": int}
    """
    total_f1 = 0.0
    total_em = 0
    for item in predictions:
        pred = item.get("llm_answer") or item.get("answer", "")
        f1, _, _ = f1_score(pred, item["ground_truth"])
        em = exact_match_score(pred, item["ground_truth"])
        total_f1 += f1
        total_em += em

    n = len(predictions)
    return {
        "avg_f1": total_f1 / n if n else 0.0,
        "avg_em": total_em / n if n else 0.0,
        "total": n,
    }
