"""
Evaluate Regular RAG (semantic only) vs T-RAG (semantic + temporal).

Two ways to run:
  1) CLI:          python evaluate.py
  2) Programmatic: from evaluate import compare_rag_vs_trag

Metrics reported:
  - avg_age_days   : mean publication age of retrieved articles
  - recency_score  : mean e^(-0.05 * age_days) — higher = more recent
  - semantic_score : mean cosine similarity
  - result_overlap : Jaccard overlap between RAG and T-RAG result sets
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict
import pandas as pd

from data_collector import collect_data
from indexer import build_index, trag_retrieve


def _age_days(pub_str: str) -> float:
    try:
        pub = datetime.strptime(pub_str[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        try:
            pub = datetime.strptime(pub_str[:10], "%Y-%m-%d")
        except Exception:
            return 0.0
    return max((datetime.utcnow() - pub).days, 0)


def metrics_for_results(results: List[Dict]) -> Dict:
    if not results:
        return {"n": 0, "avg_age_days": 0, "median_age_days": 0,
                "recency_score": 0, "semantic_score": 0}
    import math
    ages = [_age_days(r["published_at"]) for r in results]
    sem  = [r.get("semantic_score", r.get("score", 0)) for r in results]
    rec  = [math.exp(-0.05 * a) for a in ages]
    return {
        "n": len(results),
        "avg_age_days":    round(sum(ages) / len(ages), 2),
        "median_age_days": round(sorted(ages)[len(ages) // 2], 2),
        "recency_score":   round(sum(rec) / len(rec), 4),
        "semantic_score":  round(sum(sem) / len(sem), 4),
    }


def overlap(a: List[Dict], b: List[Dict]) -> float:
    sa = {r["title"] for r in a}
    sb = {r["title"] for r in b}
    if not sa and not sb:
        return 0.0
    return round(len(sa & sb) / len(sa | sb), 3)


def compare_rag_vs_trag(
    company_name: str,
    ticker: str,
    queries: List[str],
    days: int = 30,
    top_k: int = 5,
    alpha_trag: float = 0.7,
    lam: float = 0.05,
):
    """Run both retrievers on the same queries. Returns (summary_df, detail_list)."""
    print(f"\nFetching data for {company_name} ({ticker})...")
    data = collect_data(company_name, ticker, days)

    all_docs = (
        data.get("news", []) +
        data.get("reports", []) +
        data.get("social", [])
    )
    if not all_docs:
        raise RuntimeError("No documents returned — set ALPHA_VANTAGE_API_KEY in .env first")

    index, metadata, embeddings = build_index(all_docs)

    summary_rows = []
    detail_dfs   = []

    for q in queries:
        rag  = trag_retrieve(q, index, metadata, embeddings, top_k=top_k, alpha=1.0,        lam=lam)
        trag = trag_retrieve(q, index, metadata, embeddings, top_k=top_k, alpha=alpha_trag, lam=lam)

        m_rag  = metrics_for_results(rag)
        m_trag = metrics_for_results(trag)
        ov     = overlap(rag, trag)

        summary_rows.append({
            "query":            q,
            "RAG_avg_age_days": m_rag["avg_age_days"],
            "TRAG_avg_age_days":m_trag["avg_age_days"],
            "Δ_age_days":       round(m_rag["avg_age_days"] - m_trag["avg_age_days"], 2),
            "RAG_recency":      m_rag["recency_score"],
            "TRAG_recency":     m_trag["recency_score"],
            "RAG_semantic":     m_rag["semantic_score"],
            "TRAG_semantic":    m_trag["semantic_score"],
            "result_overlap":   ov,
        })

        detail = pd.DataFrame([
            {
                "rank": i + 1,
                "RAG_date":   rag[i]["published_at"][:10] if i < len(rag)  else "",
                "RAG_src":    rag[i].get("data_source", "news")[:6] if i < len(rag)  else "",
                "RAG_title":  (rag[i]["title"][:55] + "…") if i < len(rag)  and len(rag[i]["title"])  > 55 else (rag[i]["title"]  if i < len(rag)  else ""),
                "TRAG_date":  trag[i]["published_at"][:10] if i < len(trag) else "",
                "TRAG_src":   trag[i].get("data_source", "news")[:6] if i < len(trag) else "",
                "TRAG_title": (trag[i]["title"][:55] + "…") if i < len(trag) and len(trag[i]["title"]) > 55 else (trag[i]["title"] if i < len(trag) else ""),
            }
            for i in range(top_k)
        ])
        detail_dfs.append((q, detail))

    return pd.DataFrame(summary_rows), detail_dfs


if __name__ == "__main__":
    QUERIES = [
        "How did the most recent earnings affect the stock price?",
        "What are analysts saying about future growth?",
        "Any recent news about product launches or partnerships?",
        "How has investor sentiment changed in the past month?",
    ]
    summary, details = compare_rag_vs_trag(
        company_name="Apple",
        ticker="AAPL",
        queries=QUERIES,
        days=180,
        top_k=5,
        alpha_trag=0.7,
    )
    import pandas as pd
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    for q, d in details:
        print(f"\n--- {q} ---")
        print(d.to_string(index=False))
