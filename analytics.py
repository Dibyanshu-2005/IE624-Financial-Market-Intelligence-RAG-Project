"""
Analytics: correlate sentiment timelines and detected events with stock price moves.

Key outputs (all serializable for JSON / Streamlit):
  - sentiment_vs_price  : list of {date, mean_sentiment, close, return_1d, has_news}
  - event_impact        : per-event next-day / 5-day return + volatility
  - sentiment_price_corr: Pearson correlation of mean daily sentiment vs returns
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import statistics


def _stock_by_date(stock: List[Dict]) -> Dict[str, Dict]:
    return {row["Date"]: row for row in stock}


def _trading_days(stock: List[Dict]) -> List[str]:
    return sorted(row["Date"] for row in stock)


def _next_trading_day(date: str, days_idx: List[str], offset: int = 1) -> Optional[str]:
    later = [d for d in days_idx if d > date]
    if len(later) < offset:
        return None
    return later[offset - 1]


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    try:
        return round(statistics.correlation(xs, ys), 4)
    except Exception:
        return None


def merge_sentiment_with_prices(daily_sentiment: List[Dict], stock: List[Dict]) -> List[Dict]:
    """Join daily sentiment with stock prices on date.

    Carries the most recent sentiment forward to fill trading days without news.
    `has_news` flag distinguishes days with actual news from carry-forward rows.
    """
    if not stock:
        return []

    sent_by_date = {row["date"]: row for row in daily_sentiment}
    sent_dates = sorted(sent_by_date.keys())
    rows = []

    last_seen = None
    for s in stock:
        date = s["Date"]
        applicable = [d for d in sent_dates if d <= date]
        if applicable:
            last_seen = sent_by_date[applicable[-1]]
        has_news = date in sent_by_date
        rows.append({
            "date": date,
            "close": round(float(s.get("Close", 0)), 4),
            "return_1d": round(float(s.get("Return", 0)), 4),
            "mean_sentiment": last_seen["mean_sentiment"] if last_seen else None,
            "n_articles": last_seen["n_articles"] if last_seen else 0,
            "has_news": has_news,
        })
    return rows


def event_impact(events: List[Dict], stock: List[Dict]) -> List[Dict]:
    """For each event compute next-day return, 5-day cumulative return, 5-day volatility."""
    if not stock:
        return []

    by_date = _stock_by_date(stock)
    days_idx = _trading_days(stock)
    out = []

    for ev in events:
        d = ev["date"]
        same_or_after = [td for td in days_idx if td >= d]
        if not same_or_after:
            continue
        anchor = same_or_after[0]

        next_d = _next_trading_day(anchor, days_idx, offset=1)
        ret_next = float(by_date[next_d]["Return"]) if next_d else None

        future_5 = []
        idx = days_idx.index(anchor)
        for k in range(idx, min(idx + 5, len(days_idx))):
            r = by_date[days_idx[k]].get("Return", 0)
            future_5.append(float(r))

        cum_5d = round(sum(future_5), 4) if future_5 else None
        vol_5d = round(statistics.pstdev(future_5), 4) if len(future_5) > 1 else None

        out.append({
            **ev,
            "anchor_date": anchor,
            "return_next_day": round(ret_next, 4) if ret_next is not None else None,
            "return_5d": cum_5d,
            "volatility_5d": vol_5d,
        })
    return out


def sentiment_price_correlation(daily_sentiment: List[Dict], stock: List[Dict]) -> Dict:
    """Pearson correlation between mean daily sentiment and stock returns.

    Uses actual news publication dates mapped to their nearest trading day —
    avoids the zero-variance problem that occurs when carry-forward sentiment
    is used across all trading days.
    """
    if not daily_sentiment or not stock:
        return {"same_day": None, "next_day": None, "n": 0}

    stock_by_date = {s["Date"]: s for s in stock}
    days_idx = sorted(s["Date"] for s in stock)

    rows = []
    seen_trade_dates: set = set()

    for sent_row in sorted(daily_sentiment, key=lambda x: x["date"]):
        date = sent_row["date"]
        # Map news date → nearest trading day on or after it
        same_or_after = [d for d in days_idx if d >= date]
        if not same_or_after:
            continue
        trade_date = same_or_after[0]
        # Skip if another news date already mapped to this trading day
        if trade_date in seen_trade_dates:
            continue
        seen_trade_dates.add(trade_date)

        s = stock_by_date[trade_date]
        rows.append({
            "sentiment": sent_row["mean_sentiment"],
            "return_1d": float(s.get("Return", 0)),
            "trade_date": trade_date,
        })

    if len(rows) < 3:
        return {"same_day": None, "next_day": None, "n": len(rows)}

    sentiments = [r["sentiment"] for r in rows]
    returns_same = [r["return_1d"] for r in rows]
    sentiments_next = sentiments[:-1]
    returns_next = [rows[i + 1]["return_1d"] for i in range(len(rows) - 1)]

    return {
        "same_day": _pearson(sentiments, returns_same),
        "next_day": _pearson(sentiments_next, returns_next),
        "n": len(rows),
    }


def aggregate_event_type_impact(impacted_events: List[Dict]) -> List[Dict]:
    """Group events by type, return mean next-day return + count."""
    by_type: Dict[str, list] = {}
    for ev in impacted_events:
        if ev.get("return_next_day") is None:
            continue
        by_type.setdefault(ev["type"], []).append(ev["return_next_day"])

    rows = []
    for t, returns in by_type.items():
        rows.append({
            "type": t,
            "type_display": next(
                (e["type_display"] for e in impacted_events if e["type"] == t),
                t,
            ),
            "count": len(returns),
            "mean_next_day_return": round(sum(returns) / len(returns), 4),
            "min": round(min(returns), 4),
            "max": round(max(returns), 4),
        })
    rows.sort(key=lambda r: abs(r["mean_next_day_return"]), reverse=True)
    return rows
