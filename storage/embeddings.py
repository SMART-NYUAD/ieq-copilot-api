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

# bge-*-en-v1.5 is trained with an ASYMMETRIC retrieval objective: the query side carries
# an instruction prefix and the passage side does not. Embedding both the same way is the
# documented misuse, and it costs ranking quality most on short questions — which is
# exactly what this system receives ("what is CO2?"). Measure it with
# `python tests/retrieval_eval.py --ablate` before changing this string.
#
# The prefix belongs ONLY on queries. Stored card/guideline vectors must stay un-prefixed,
# so adding this does not invalidate anything already embedded — no re-embedding needed.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _query_instruction(model_name: str) -> str:
    """The instruction prefix this model expects on the query side, or ""."""
    return _BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""


def embed_texts(texts: List[str], model_name: str = _DEFAULT_MODEL) -> List[List[float]]:
    """Embed PASSAGES (documents) for storage — no instruction prefix.

    Use :func:`embed_query` for the search side. The two are not interchangeable for bge
    models; see ``_BGE_QUERY_INSTRUCTION``.
    """
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



def embed_query(text: str, model_name: str = _DEFAULT_MODEL) -> List[float]:
    """Embed one SEARCH QUERY, applying the model's instruction prefix.

    Every retrieval path should call this rather than :func:`embed_texts`, so the query
    side stays consistent no matter which store is being searched.
    """
    vectors = embed_texts([_query_instruction(model_name) + str(text or "")], model_name)
    return vectors[0] if vectors else []


def embed_documents(texts: List[str], model_name: str = _DEFAULT_MODEL) -> List[List[float]]:
    """Explicit alias for the passage side, so call sites state which side they are on."""
    return embed_texts(texts, model_name)


def embedding_dimension(model_name: str = _DEFAULT_MODEL) -> int:
    """The model's native vector width — the value a schema column should declare."""
    vectors = embed_texts(["dimension probe"], model_name)
    return len(vectors[0]) if vectors else 0
