"""
FAISS-based T-RAG retrieval with on-disk persistence and lazy model loading.

Core scoring:
    Score(d, q) = α · cosine(q, d) + (1 − α) · e^(−λ · Δt_days)

Performance notes:
  - SentenceTransformer is lazy-loaded on first use (not at import time)
  - build_index() caches index + metadata + embeddings to disk; subsequent
    calls with the same article set skip re-embedding entirely
"""

from __future__ import annotations

import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_MODEL: Optional[SentenceTransformer] = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    return _model().encode(texts, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _key_for(articles: List[Dict]) -> str:
    parts = []
    for a in articles[:200]:
        parts.append(a.get("title", "") + "|" + a.get("published_at", ""))
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def _paths(key: str):
    return (
        CACHE_DIR / f"index_{key}.faiss",
        CACHE_DIR / f"meta_{key}.pkl",
        CACHE_DIR / f"emb_{key}.npy",
    )


def _save_index(index, metadata: List[Dict], embeddings: np.ndarray, key: str) -> None:
    p_idx, p_meta, p_emb = _paths(key)
    try:
        faiss.write_index(index, str(p_idx))
        with open(p_meta, "wb") as f:
            pickle.dump(metadata, f)
        np.save(p_emb, embeddings)
    except Exception as e:
        print(f"  [indexer] cache write failed: {e}")


def _load_index(key: str):
    p_idx, p_meta, p_emb = _paths(key)
    if not (p_idx.exists() and p_meta.exists() and p_emb.exists()):
        return None
    try:
        index = faiss.read_index(str(p_idx))
        with open(p_meta, "rb") as f:
            metadata = pickle.load(f)
        embeddings = np.load(p_emb)
        return index, metadata, embeddings
    except Exception as e:
        print(f"  [indexer] cache load failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Build / retrieve
# ---------------------------------------------------------------------------
def build_index(articles: List[Dict], persist: bool = True):
    """Build a FAISS index from article dicts, with disk caching.

    Preserves url, sentiment, topics, data_source in metadata so downstream
    tabs (Q&A, sentiment, events) can access them without re-fetching.
    """
    if not articles:
        raise ValueError("build_index: empty article list")

    key = _key_for(articles)

    if persist:
        cached = _load_index(key)
        if cached is not None:
            print(f"  [indexer] loaded cached index ({len(cached[1])} docs)")
            return cached

    texts = [
        f"{a.get('title', '')}. {a.get('description', '')}" for a in articles
    ]
    print(f"  [indexer] embedding {len(texts)} documents…")
    embeddings = np.array(embed_texts(texts)).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    metadata = []
    for a in articles:
        metadata.append({
            "title":                a.get("title", ""),
            "description":          a.get("description", ""),
            "published_at":         a.get("published_at", ""),
            "source":               a.get("source", ""),
            "url":                  a.get("url", ""),
            "topics":               a.get("topics", []),
            "data_source":          a.get("data_source", "news"),
            "sentiment":            a.get("sentiment"),
            "av_overall_sentiment": a.get("av_overall_sentiment"),
            "av_ticker_sentiment":  a.get("av_ticker_sentiment"),
        })

    if persist:
        _save_index(index, metadata, embeddings, key)
        print(f"  [indexer] index cached ({len(metadata)} docs)")

    return index, metadata, embeddings


def time_decay(published_at_str: str, reference_time=None, lam: float = 0.05) -> float:
    if reference_time is None:
        reference_time = datetime.utcnow()
    try:
        pub_time = datetime.strptime(published_at_str[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        try:
            pub_time = datetime.strptime(published_at_str[:10], "%Y-%m-%d")
        except Exception:
            pub_time = reference_time
    delta_days = max((reference_time - pub_time).days, 0)
    return float(np.exp(-lam * delta_days))


def trag_retrieve(
    query: str,
    index,
    metadata: List[Dict],
    embeddings: np.ndarray,
    top_k: int = 5,
    alpha: float = 0.7,
    lam: float = 0.05,
) -> List[Dict]:
    """T-RAG retrieval: Score = α·semantic + (1−α)·temporal decay."""
    if index is None or not metadata:
        return []

    q_emb = np.array(embed_texts([query])).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, len(metadata))
    scores  = scores[0]
    indices = indices[0]

    scored = []
    for rank, idx in enumerate(indices):
        if idx == -1:
            continue
        sem  = float(scores[rank])
        temp = time_decay(metadata[idx]["published_at"], lam=lam)
        combined = alpha * sem + (1 - alpha) * temp
        scored.append((combined, sem, temp, idx))

    scored.sort(reverse=True, key=lambda x: x[0])

    out = []
    for combined, sem, temp, idx in scored[:top_k]:
        m = metadata[idx]
        out.append({
            "score":          round(combined, 4),
            "semantic_score": round(sem, 4),
            "temporal_score": round(temp, 4),
            "title":          m["title"],
            "description":    m["description"],
            "published_at":   m["published_at"],
            "source":         m["source"],
            "url":            m.get("url", ""),
            "data_source":    m.get("data_source", "news"),
            "sentiment":      m.get("sentiment"),
        })
    return out
