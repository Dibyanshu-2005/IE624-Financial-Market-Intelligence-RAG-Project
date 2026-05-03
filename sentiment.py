"""
Sentiment analysis module — FinBERT (financial-domain BERT) + VADER fallback.

Each article gets a `sentiment` block with:
  - finbert_score   : signed score in [-1, +1]
  - finbert_label   : "positive" / "neutral" / "negative"
  - vader_score     : signed compound score in [-1, +1]
  - av_score        : Alpha Vantage's per-ticker score (if present)
  - ensemble_score  : weighted average across whichever signals are available
  - ensemble_label  : derived from ensemble_score

The first FinBERT load downloads ~440 MB.  We fall back gracefully to VADER
if torch / transformers are unavailable.
"""

from __future__ import annotations

from typing import List, Dict, Optional

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

_FINBERT = None
_FINBERT_FAILED = False


def _load_finbert():
    global _FINBERT, _FINBERT_FAILED
    if _FINBERT is not None or _FINBERT_FAILED:
        return _FINBERT
    try:
        from transformers import pipeline
        _FINBERT = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
            top_k=None,
        )
    except Exception as e:
        print(f"[sentiment] FinBERT unavailable, falling back to VADER: {e}")
        _FINBERT_FAILED = True
        _FINBERT = None
    return _FINBERT


def _label_from_score(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def _vader_score(text: str) -> float:
    if _VADER is None or not text:
        return 0.0
    return _VADER.polarity_scores(text).get("compound", 0.0)


def _finbert_score(text: str) -> Optional[Dict]:
    pipe = _load_finbert()
    if pipe is None or not text:
        return None
    try:
        out = pipe(text[:1500])
    except Exception as e:
        print(f"[sentiment] finbert call failed: {e}")
        return None

    if isinstance(out, list) and out and isinstance(out[0], list):
        scores = {d["label"].lower(): d["score"] for d in out[0]}
    elif isinstance(out, list) and out and isinstance(out[0], dict):
        scores = {out[0]["label"].lower(): out[0]["score"]}
    else:
        return None

    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    signed = pos - neg
    return {"score": signed, "label": _label_from_score(signed)}


def score_article(article: Dict) -> Dict:
    """Return a sentiment dict for a single article.  Pure function."""
    text = f"{article.get('title','')}. {article.get('description','')}".strip()

    fb = _finbert_score(text)
    vd = _vader_score(text)
    av = article.get("av_ticker_sentiment")
    if av is None:
        av = article.get("av_overall_sentiment")

    parts, weights = [], []
    if fb is not None:
        parts.append(fb["score"]); weights.append(0.55)
    if av is not None:
        parts.append(float(av));   weights.append(0.30)
    if _VADER is not None:
        parts.append(vd);          weights.append(0.15)

    if parts:
        total_w = sum(weights)
        ensemble = sum(p * w for p, w in zip(parts, weights)) / total_w
    else:
        ensemble = 0.0

    return {
        "finbert_score": round(fb["score"], 4) if fb else None,
        "finbert_label": fb["label"] if fb else None,
        "vader_score": round(vd, 4),
        "av_score": round(float(av), 4) if av is not None else None,
        "ensemble_score": round(ensemble, 4),
        "ensemble_label": _label_from_score(ensemble),
    }


def score_articles(articles: List[Dict], use_finbert: bool = True) -> List[Dict]:
    """Add `sentiment` field to every article in place, return the list."""
    if not use_finbert:
        global _FINBERT_FAILED
        _FINBERT_FAILED = True

    for a in articles:
        a["sentiment"] = score_article(a)
    return articles


def aggregate_daily(articles: List[Dict]) -> List[Dict]:
    """Bucket articles by date, return [{date, mean_sentiment, n, pos_n, neg_n, neu_n}]."""
    by_day: Dict[str, list] = {}
    for a in articles:
        date = a.get("published_at", "")[:10]
        if not date:
            continue
        s = a.get("sentiment", {})
        score = s.get("ensemble_score")
        if score is None:
            continue
        by_day.setdefault(date, []).append(s)

    rows = []
    for date in sorted(by_day):
        scores = [x["ensemble_score"] for x in by_day[date]]
        labels = [x["ensemble_label"] for x in by_day[date]]
        rows.append({
            "date": date,
            "mean_sentiment": round(sum(scores) / len(scores), 4),
            "n_articles": len(scores),
            "pos_n": sum(1 for l in labels if l == "positive"),
            "neg_n": sum(1 for l in labels if l == "negative"),
            "neu_n": sum(1 for l in labels if l == "neutral"),
        })
    return rows
