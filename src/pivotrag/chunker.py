"""Document loading and chunking — no external pipeline dependency."""

from __future__ import annotations

import logging
from pathlib import Path

import tiktoken

from pivotrag.config import CHUNK_OVERLAP, CHUNK_SIZE
from pivotrag.models import ChunkNode

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".text", ".markdown"})


def load_documents(input_path: str) -> list[dict]:
    """Load documents from a file or directory.

    Args:
        input_path: 文件路径、目录路径、或包含 input/ 子目录的根路径。

    Returns:
        [{"title": str, "text": str, "source": str}, ...]
    """
    path = Path(input_path)

    if path.is_file():
        return [_load_file(path)]

    if path.is_dir():
        # 优先查找 input/ 子目录
        sub = path / "input"
        search_dir = sub if sub.is_dir() else path

        docs: list[dict] = []
        for fp in sorted(search_dir.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.append(_load_file(fp))
        return docs

    raise FileNotFoundError(f"Input path not found: {input_path}")


def _load_file(filepath: Path) -> dict:
    """读取单个文本文件."""
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    return {"title": filepath.stem, "text": text, "source": str(filepath)}


def chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    encoding_name: str = "cl100k_base",
) -> list[ChunkNode]:
    """将文档按 token 数分块.

    Args:
        documents: [{"title", "text", "source"}, ...]
        chunk_size: 每块最大 token 数
        chunk_overlap: 相邻块重叠 token 数
        encoding_name: tiktoken 编码名

    Returns:
        List[ChunkNode]
    """
    encoder = tiktoken.get_encoding(encoding_name)
    chunks: list[ChunkNode] = []

    for doc in documents:
        tokens = encoder.encode(doc["text"])
        if not tokens:
            continue

        doc_id = doc["title"]
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoder.decode(chunk_tokens)

            chunks.append(
                ChunkNode(
                    id=f"{doc_id}_{idx}",
                    text=chunk_text,
                    n_tokens=len(chunk_tokens),
                    document_ids=[doc_id],
                )
            )
            idx += 1
            next_start = start + chunk_size - chunk_overlap
            if next_start >= len(tokens):
                break
            start = next_start

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
