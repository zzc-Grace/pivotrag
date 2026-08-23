"""Runtime configuration loaded from environment variables.

No credentials or provider-specific endpoints are stored in the source tree.
Set the ``PIVORAG_*`` variables in the shell (or in a local, untracked
``.env`` file) before running an operation that calls a remote model service.
"""

import os


def _env(name: str, default: str = "") -> str:
    """Read and trim an environment variable without exposing credentials."""
    return os.getenv(name, default).strip()

# ── LLM ──
LLM_API_KEY: str = _env("PIVORAG_LLM_API_KEY")
LLM_BASE_URL: str = _env("PIVORAG_LLM_BASE_URL")
LLM_MODEL: str = _env("PIVORAG_LLM_MODEL", "your-llm-model")

# ── Embedding ──
EMBEDDING_API_KEY: str = _env("PIVORAG_EMBEDDING_API_KEY")
EMBEDDING_BASE_URL: str = _env("PIVORAG_EMBEDDING_BASE_URL")
EMBEDDING_MODEL: str = _env("PIVORAG_EMBEDDING_MODEL", "your-embedding-model")
EMBEDDING_DIM: int = 1024
EMBEDDING_BATCH_SIZE: int = 16

# ── Reranker ──
RERANK_API_KEY: str = _env("PIVORAG_RERANK_API_KEY")
RERANK_BASE_URL: str = _env("PIVORAG_RERANK_BASE_URL")
RERANK_MODEL: str = _env("PIVORAG_RERANK_MODEL", "your-reranker-model")
USE_RERANKER: bool = os.getenv("PIVORAG_USE_RERANKER", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
NO_RERANK_VECTOR_K: int = int(os.getenv("PIVORAG_NO_RERANK_VECTOR_K", "5"))

# ── 文档分块 ──
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

# ── 实体抽取 ──
ENTITY_EXTRACTOR_BACKEND: str = os.getenv(
    "PIVORAG_ENTITY_EXTRACTOR_BACKEND", "spacy"
)  # "spacy" or "llm"
SPACY_MODEL_PATH: str = os.getenv(
    "PIVORAG_SPACY_MODEL_PATH", "en_core_web_lg"
)
SPACY_ENTITY_TYPES: frozenset[str] = frozenset(
    os.getenv(
        "PIVORAG_SPACY_ENTITY_TYPES",
        "PERSON,ORG,GPE,LOC,FAC,PRODUCT,EVENT,WORK_OF_ART,LAW,LANGUAGE,NORP",
    ).split(",")
)
SPACY_BATCH_SIZE: int = int(os.getenv("PIVORAG_SPACY_BATCH_SIZE", "64"))

ENTITY_DESC_MAX_LEN: int = 100
MAX_CONCURRENT_LLM_CALLS: int = 10
EXTRACT_MAX_RETRIES: int = 3

# ── 图构建 ──
SIMILARITY_THRESHOLD: float = 0.8  # 构建"相似边"（similarity edge）的相似度阈值。
ENTITY_MERGE_SIM_THRESHOLD: float = 0.8  # 合并图中实体节点时的相似度阈值。
E_SIM_TOP_L: int = 50  # ANN 搜索时每个实体的最近邻候选数。

# ── 检索 ──
K1_GRAPH: int = 10
K2_VECTOR: int = 10
K3_FINAL: int = 10
PPR_ALPHA: float = 0.85
PPR_MODE: str = os.getenv("PIVORAG_PPR_MODE", "degree_normalized")  # "original" or "degree_normalized"
PPR_MAX_ITER: int = 20
PPR_TOLERANCE: float = 1e-4
QUERY_TOP_N_ENTITIES: int = 5  # 从用户问题中最多抽取的实体数量。
ENTITY_MATCH_TOP_N: int = 5  # 每个查询实体匹配到的种子实体数量。
BETA: float = 0.5  # Query gate 强度
PPR_LOCAL_HOP: int = 2  # 局部子图跳数
LAMBDA_STR: float = 0.5  # 结构分数融合权重

# ── 查询分解 ──
QUERY_DECOMPOSE: bool = os.getenv(
    "PIVORAG_QUERY_DECOMPOSE", "false"
).strip().lower() in {"1", "true", "yes", "on"}
DECOMPOSE_MAX_SUB_QUERIES: int = 4    # 最多分解为几个子问题
