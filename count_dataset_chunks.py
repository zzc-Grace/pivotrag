"""Count sentence-level chunks in a context-based QA JSON dataset.

In this dataset, each context item has the form ``[title, [paragraph, ...]]``.
Every element of the inner list is counted as one chunk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def count_chunks(data: list[dict[str, Any]]) -> int:
    """Return the total number of sentence-level chunks in ``context`` fields."""
    total = 0
    for sample_index, item in enumerate(data):
        context = item.get("context", [])
        if not isinstance(context, list):
            raise ValueError(f"Sample {sample_index}: 'context' must be a list.")

        for context_index, document in enumerate(context):
            if not (
                isinstance(document, list)
                and len(document) == 2
                and isinstance(document[1], list)
            ):
                raise ValueError(
                    f"Sample {sample_index}, context {context_index}: "
                    "expected [title, [chunk, ...]]."
                )
            total += len(document[1])
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count chunks in a context-based QA JSON file."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the dataset JSON file.",
    )
    args = parser.parse_args()

    with args.input_file.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("The dataset root must be a JSON list.")

    print(f"Samples: {len(data)}")
    print(f"Chunks: {count_chunks(data)}")


if __name__ == "__main__":
    main()
