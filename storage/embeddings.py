"""Embedding helpers used by knowledge-card retrieval paths."""

from __future__ import annotations

from typing import List

try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


_EMBEDDING_MODEL_CACHE: dict[str, "SentenceTransformer"] = {}
_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def embed_texts(texts: List[str], model_name: str = _DEFAULT_MODEL) -> List[List[float]]:
    """Generate normalized vectors for semantic retrieval."""
    if not texts:
        return []
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers package not available. Install with: pip install sentence-transformers"
        )

    try:
        if model_name not in _EMBEDDING_MODEL_CACHE:
            _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        model = _EMBEDDING_MODEL_CACHE[model_name]
        embeddings = model.encode(
            texts,
            convert_to_numpy=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc

    result: List[List[float]] = []
    for emb in embeddings:
        if hasattr(emb, "tolist"):
            result.append(emb.tolist())
        elif isinstance(emb, list):
            result.append(emb)
        else:
            result.append(list(emb))
    return result

