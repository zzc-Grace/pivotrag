"""使用LLM评估答案一致性."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pivotrag.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from pivotrag.http_client import chat_completion

logger = logging.getLogger(__name__)

# ── 评估提示词 ──

_EVAL_SYSTEM_PROMPT = """\
You are an expert evaluator for question-answering systems.
Your task is to determine if the LLM's answer captures the core information requested by the question, compared to the ground truth.

Question: {question}

LLM Answer: {llm_answer}

Ground Truth: {ground_truth}

## Evaluation Rules

Mark as CORRECT if ANY of the following hold:
1. The LLM answer is semantically equivalent to the ground truth (same meaning, different wording).
2. The LLM answer contains the ground truth as a substring, or the ground truth contains the LLM answer as a substring — as long as the core answer is preserved.
3. The LLM answer is more specific or more general than the ground truth, but still correctly answers the question. For example, if ground truth is "Imperial Russian troops" and LLM says "Russian troops", that is correct.
4. The LLM answer provides the key entity/fact that the question asks for, even if it omits secondary details present in the ground truth. For example, if ground truth is "the Slavic women accompanying their husbands in the First Balkan War" and LLM says "First Balkan War", that is correct because the core event is captured.
5. The LLM answer is a valid rephrasing, abbreviation, or expansion of the ground truth. For example: "1970s and 1980s" = "the 1970s and 1980s", "a minor basilica" = "minor basilica".

Mark as INCORRECT ONLY if:
1. The LLM answer refers to a completely different entity, fact, or time period than the ground truth.
2. The LLM answer contradicts the ground truth.
3. The LLM answer is "Cannot be answered" or empty when the ground truth provides a specific answer.
4. The LLM answer misses the core information that the question asks for. For example, if the question asks "What award did X receive?" and the LLM says "Academy Award for Best Original Song" but the ground truth is "Academy Award for Best Supporting Actor" — these are different awards, so it is incorrect.

## Examples

Q: "What years did the cartel operate?"
LLM: "1970s and 1980s"
GT: "1970s and 1980s"
→ CORRECT (exact match)

Q: "What military overran the city?"
LLM: "Imperial Russian troops"
GT: "Russian troops"
→ CORRECT (LLM is more specific but contains the core answer)

Q: "What was the march written in honor of?"
LLM: "First Balkan War"
GT: "the Slavic women accompanying their husbands in the First Balkan War"
→ CORRECT (LLM captures the core event, omits secondary detail)

Q: "What is the status of the building?"
LLM: "minor basilica"
GT: "a minor basilica"
→ CORRECT (trivial wording difference)

Q: "What award did the singer receive?"
LLM: "Academy Award for Best Original Song"
GT: "Academy Award for Best Supporting Actor"
→ INCORRECT (different awards entirely)

Q: "What team does the player belong to?"
LLM: "Arizona Cardinals"
GT: "New England Patriots"
→ INCORRECT (different teams)

Q: "Where was he born?"
LLM: "Cannot be answered based on the provided context."
GT: "Clatskanie"
→ INCORRECT (LLM failed to answer)

Output your evaluation in the following JSON format:
{{
  "is_correct": true/false,
  "reason": "Brief explanation of why the answer is correct or incorrect"
}}

Evaluation Result (JSON only):"""


# ── 评估函数 ──

async def evaluate_single_answer(
    question: str,
    llm_answer: str,
    ground_truth: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """使用LLM评估单个答案的一致性.

    Args:
        question: 问题
        llm_answer: LLM生成的答案
        ground_truth: 标准答案
        semaphore: 并发控制信号量

    Returns:
        包含评估结果的字典
    """
    async with semaphore:
        prompt = _EVAL_SYSTEM_PROMPT.format(
            question=question,
            llm_answer=llm_answer,
            ground_truth=ground_truth,
        )

        try:
            content, _ = await chat_completion(
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful evaluation assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )

            # 尝试解析JSON响应
            # 清理可能的markdown代码块
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                result = json.loads(content)
                is_correct = bool(result.get("is_correct", False))
                reason = str(result.get("reason", "No reason provided"))
            except json.JSONDecodeError:
                # 如果JSON解析失败，根据内容推断
                content_lower = content.lower()
                is_correct = "true" in content_lower and "false" not in content_lower.replace("true", "")
                reason = f"Failed to parse JSON. Raw response: {content[:200]}"

            return {
                "is_correct": is_correct,
                "reason": reason,
                "raw_response": content,
            }

        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return {
                "is_correct": False,
                "reason": f"Error during evaluation: {str(e)}",
                "raw_response": "",
            }


# ── 批量评估 ──

async def batch_evaluate_answers(
    input_file: str,
    output_file: str,
    max_concurrency: int = 10,
) -> dict[str, Any]:
    """批量评估答案一致性.

    Args:
        input_file: 输入的eval_result.json文件路径
        output_file: 输出结果文件路径
        max_concurrency: 最大并发数

    Returns:
        评估统计结果
    """
    # 加载输入数据
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        logger.error("No predictions found in input file")
        return {"error": "No predictions found"}

    logger.info(f"Evaluating {len(predictions)} answers...")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_item(item: dict, index: int) -> dict:
        """处理单个评估项."""
        question = item.get("question", "")
        llm_answer = item.get("llm_answer", "")
        ground_truth = item.get("ground_truth", "")

        eval_result = await evaluate_single_answer(
            question, llm_answer, ground_truth, semaphore
        )

        return {
            "id": item.get("id", f"item_{index}"),
            "question": question,
            "llm_answer": llm_answer,
            "ground_truth": ground_truth,
            "is_correct": eval_result["is_correct"],
            "reason": eval_result["reason"],
            "raw_response": eval_result["raw_response"],
        }

    # 批量处理
    t_start = time.perf_counter()
    tasks = [_process_item(p, i) for i, p in enumerate(predictions)]
    results = await asyncio.gather(*tasks)
    t_end = time.perf_counter()

    # 计算统计信息
    total = len(results)
    correct_count = sum(1 for r in results if r["is_correct"])
    accuracy = correct_count / total if total > 0 else 0.0

    summary = {
        "total": total,
        "correct": correct_count,
        "incorrect": total - correct_count,
        "accuracy": round(accuracy, 4),
        "eval_time_seconds": round(t_end - t_start, 3),
    }

    # 保存结果
    output_data = {
        "summary": summary,
        "evaluations": results,
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Evaluation complete: {correct_count}/{total} correct (accuracy={accuracy:.4f}), "
        f"time={t_end - t_start:.3f}s"
    )
    logger.info(f"Results saved to {output_file}")

    return summary


# ── CLI ──

async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python llm_evaluate.py <input_file> [output_file] [max_concurrency]")
        print("")
        print("Arguments:")
        print("  input_file: Path to eval_result.json")
        print("  output_file: Path to save evaluation results (default: llm_eval_result.json)")
        print("  max_concurrency: Maximum concurrent LLM calls (default: 10)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "llm_eval_result.json"
    max_concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    summary = await batch_evaluate_answers(input_file, output_file, max_concurrency)
    print("\nEvaluation Summary:")
    print(f"  Total: {summary['total']}")
    print(f"  Correct: {summary['correct']}")
    print(f"  Incorrect: {summary['incorrect']}")
    print(f"  Accuracy: {summary['accuracy']:.4f}")
    print(f"  Time: {summary['eval_time_seconds']:.3f}s")


if __name__ == "__main__":
    asyncio.run(_main())
