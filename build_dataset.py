"""Import a QA dataset, build an index, and optionally evaluate it.

The input format is the common HotpotQA/MuSiQue-style JSON list where each
sample contains ``context`` as ``[[title, [sentence, ...]], ...]`` and the
question/answer fields.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

from pivotrag.config import EMBEDDING_MODEL, LLM_MODEL
from pivotrag.pipeline import batch_evaluate, build_index


def _export_documents(input_file: str, root_dir: str, limit: int | None = None) -> int:
    """Extract unique context documents into ``root_dir/input``."""
    input_dir = Path(root_dir) / "input"
    if input_dir.exists() and any(input_dir.iterdir()):
        raise FileExistsError(
            f"Input directory is not empty: {input_dir}. "
            "Choose a new --output directory or clear it explicitly."
        )
    input_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None and limit > 0:
        data = data[:limit]

    seen_titles: set[str] = set()

    for item in data:
        for title, sentences in item.get("context", []):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            text = " ".join(sentences)
            safe_name = re.sub(r'[<>:"/\\|?*()]', "_", str(title)).strip(" ._")
            safe_name = safe_name or "document"
            (input_dir / f"{safe_name}.txt").write_text(text, encoding="utf-8")

    return len(seen_titles)


def _export_qa(input_file: str, root_dir: str, limit: int | None = None) -> int:
    """Prepare the question/answer fields for PivoRAG evaluation."""
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None and limit > 0:
        data = data[:limit]

    qa_pairs = [
        {
            "id": item.get("id", item.get("_id", "")),
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": item.get("supporting_facts", []),
        }
        for item in data
    ]

    output_path = Path(root_dir) / "qa.json"
    output_path.write_text(
        json.dumps(qa_pairs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(qa_pairs)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate a PivoRAG dataset index")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to a context-based QA JSON file",
    )
    parser.add_argument(
        "-o", "--output",
        default="dataset_root",
        help="Root directory for index output",
    )
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run evaluation after building the index (default: true)",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit import and evaluation to the first N samples (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_file = args.input
    root_dir = args.output

    if not Path(input_file).exists():
        print(f"Error: input file not found: {input_file}")
        sys.exit(1)

    t_import_start = time.perf_counter()

    doc_count = _export_documents(input_file, root_dir, limit=args.limit)
    qa_count = _export_qa(input_file, root_dir, limit=args.limit)

    print(f"Exported {doc_count} docs and {qa_count} QA pairs to {root_dir}/")

    print("Building PivoRAG index...")
    usage = await build_index(root_dir)

    t_import_end = time.perf_counter()
    import_time_seconds = round(t_import_end - t_import_start, 3)
    print(f"  Import and index build time: {import_time_seconds:.1f}s")

    usage_report = {
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "doc_count": doc_count,
        "qa_count": qa_count,
        "import_time_seconds": import_time_seconds,
        "token_usage": usage.to_dict(),
    }
    report_path = Path(root_dir) / "token_usage.json"
    report_path.write_text(
        json.dumps(usage_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Token usage report saved to {report_path}")
    print(f"  LLM tokens: {usage.llm_input_tokens} in + {usage.llm_output_tokens} out")
    print(f"  Embedding tokens: {usage.embedding_input_tokens} in")
    print(f"  Total tokens: {usage.to_dict()['total_tokens']}")

    if args.eval:
        print("\nRunning evaluation...")
        qa_file = str(Path(root_dir) / "qa.json")
        eval_output = str(Path(root_dir) / "eval_result.json")
        metrics = await batch_evaluate(root_dir, qa_file, eval_output)
        print("\nEvaluation results:")
        print(f"  F1:        {metrics['avg_f1']:.4f}")
        print(f"  EM:        {metrics['avg_em']:.4f}")
        print(f"  Recall@5:  {metrics['avg_recall@5']:.4f}")
        print(f"  Recall@10: {metrics['avg_recall@10']:.4f}")
        print(f"  Total:     {metrics['total']}")
        print(f"  Eval time: {metrics['eval_time_seconds']:.1f}s")
        print(f"  Results saved to {eval_output}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
