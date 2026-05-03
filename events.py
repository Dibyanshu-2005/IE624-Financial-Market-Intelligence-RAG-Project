"""
Event detection module.

Classifies each article into business-relevant event types using:
  1. Alpha Vantage pre-tagged topics (high precision)
  2. Keyword/regex rules over title + description

Output per event:
    {date, type, type_display, severity, headline, source, sentiment, url, article_idx}
"""

from __future__ import annotations

import re
from typing import List, Dict
from collections import Counter


EVENT_TYPES = {
    "earnings":        {"display": "Earnings",            "color": "#1f77b4"},
    "guidance":        {"display": "Guidance Update",     "color": "#9467bd"},
    "mna":             {"display": "M&A / Acquisition",   "color": "#2ca02c"},
    "product_launch":  {"display": "Product Launch",      "color": "#17becf"},
    "regulatory":      {"display": "Regulatory / Legal",  "color": "#d62728"},
    "leadership":      {"display": "Leadership Change",   "color": "#8c564b"},
    "analyst_action":  {"display": "Analyst Action",      "color": "#e377c2"},
    "macro":           {"display": "Macro / Policy",      "color": "#7f7f7f"},
    "partnership":     {"display": "Partnership / Deal",  "color": "#bcbd22"},
    "stock_action":    {"display": "Stock Action",        "color": "#ff7f0e"},
    "other":           {"display": "Other",               "color": "#aec7e8"},
}

AV_TOPIC_MAP = {
    "earnings": "earnings",
    "ipo": "stock_action",
    "mergers_and_acquisitions": "mna",
    "financial_markets": "macro",
    "economy_fiscal": "macro",
    "economy_monetary": "macro",
    "economy_macro": "macro",
    "energy_transportation": "other",
    "finance": "other",
    "life_sciences": "other",
    "manufacturing": "other",
    "real_estate": "other",
    "retail_wholesale": "other",
    "technology": "other",
    "blockchain": "other",
}

KEYWORD_RULES = [
    # earnings
    ("earnings",       r"\b(beats?|misses?|tops?)\s+(estimates?|expectations?|forecasts?)", 0.9),
    ("earnings",       r"\b(reports?|posts?)\s+(q[1-4]|quarterly|annual)\s+(earnings|results|revenue)", 0.85),
    ("earnings",       r"\b(earnings|EPS|revenue)\s+(beat|miss|surge|jump|fall|drop)", 0.85),
    ("earnings",       r"\b(record|all[- ]time)\s+(profit|revenue|earnings)", 0.8),
    ("earnings",       r"\b10-[KQ]\b|\bForm\s+10-[KQ]\b", 0.75),

    # guidance
    ("guidance",       r"\b(raises?|lifts?|boosts?|hikes?)\s+(guidance|outlook|forecast)", 0.85),
    ("guidance",       r"\b(cuts?|lowers?|slashes?|trims?|reduces?)\s+(guidance|outlook|forecast)", 0.9),
    ("guidance",       r"\b(reaffirms?|maintains?)\s+(guidance|outlook)", 0.4),

    # M&A
    ("mna",            r"\b(acquir(es?|ing|ition)|buyout|takeover)\b", 0.9),
    ("mna",            r"\b(merger|merges?\s+with|to\s+merge)\b", 0.9),
    ("mna",            r"\b(deal|agreement)\s+to\s+(buy|acquire|purchase)", 0.85),

    # product
    ("product_launch", r"\b(launch(es|ed|ing)?|unveils?|announces?\s+new|introduces?\s+new|debuts?)\b", 0.65),
    ("product_launch", r"\b(release|releasing|releases?)\s+(new|upcoming)", 0.6),

    # regulatory / legal
    ("regulatory",     r"\b(SEC|FTC|DOJ|FDA|EU\s+Commission|antitrust|regulator|regulatory)\b", 0.7),
    ("regulatory",     r"\b(lawsuit|sued|sues|litigation|investigation|probe|fines?|penalty)\b", 0.85),
    ("regulatory",     r"\b(banned?|sanctions?|tariff|recall)\b", 0.8),
    ("regulatory",     r"\b8-K\b|\bForm\s+8-K\b", 0.6),

    # leadership
    ("leadership",     r"\b(CEO|CFO|COO|CTO|Chairman|President)\s+(steps?\s+down|resigns?|fired|appointed|named)\b", 0.9),
    ("leadership",     r"\b(new|appoints?|hires?|names?)\s+(CEO|CFO|COO|CTO|Chairman|President)\b", 0.85),

    # analyst
    ("analyst_action", r"\b(upgrades?|downgrades?)\s+(to|from)\s+(buy|sell|hold|outperform|neutral)", 0.85),
    ("analyst_action", r"\b(price target|PT)\s+(raised|cut|lowered|increased|reduced)\b", 0.7),
    ("analyst_action", r"\b(rated?|rating)\s+(buy|sell|hold|outperform|underperform)\b", 0.55),

    # partnership
    ("partnership",    r"\b(partners?\s+with|partnership|collaboration|joint\s+venture)\b", 0.6),
    ("partnership",    r"\b(signs?\s+(a\s+)?(deal|contract|agreement))\b", 0.65),

    # stock action
    ("stock_action",   r"\b(stock\s+split|share\s+buyback|buybacks?|dividend\s+(hike|raise|cut))\b", 0.75),
    ("stock_action",   r"\b(IPO|going public|secondary offering)\b", 0.8),

    # macro
    ("macro",          r"\b(Fed|Federal Reserve|interest rate|inflation|recession|GDP)\b", 0.5),
    ("macro",          r"\b(tariff|trade war|sanctions|geopolitical)\b", 0.6),
]

KEYWORD_RULES = [(et, re.compile(p, re.IGNORECASE), sev) for (et, p, sev) in KEYWORD_RULES]


def _detect_event_types(article: Dict) -> List[Dict]:
    text = f"{article.get('title','')} {article.get('description','')}"
    hits: Dict[str, float] = {}

    for topic in article.get("topics") or []:
        key = (topic or "").lower().replace(" ", "_").replace("&", "and")
        et = AV_TOPIC_MAP.get(key)
        if et:
            hits[et] = max(hits.get(et, 0), 0.7)

    for et, regex, sev in KEYWORD_RULES:
        if regex.search(text):
            hits[et] = max(hits.get(et, 0), sev)

    if not hits:
        return []  # No catch-all "other" — only emit events for actual keyword/topic matches
    return [{"type": et, "severity": round(sev, 2)} for et, sev in hits.items()]


# Minimum severity an event must have to be included (filters low-signal matches)
MIN_EVENT_SEVERITY = 0.5

def detect_events(articles: List[Dict]) -> List[Dict]:
    """Run event detection across all articles. Returns flat list sorted by date desc.

    Only emits events for articles that match actual keyword or topic rules
    (no catch-all "other"), and deduplicates same-day same-type events,
    keeping only the highest-severity article per (date, type) pair.
    """
    # First pass: collect all hits
    raw: List[Dict] = []
    for i, a in enumerate(articles):
        if a.get("data_source") == "social":
            continue
        detected = _detect_event_types(a)
        date = a.get("published_at", "")[:10]
        if not date:
            continue
        for d in detected:
            if d["severity"] < MIN_EVENT_SEVERITY:
                continue
            raw.append({
                "date":           date,
                "type":           d["type"],
                "type_display":   EVENT_TYPES[d["type"]]["display"],
                "severity":       d["severity"],
                "headline":       a.get("title", ""),
                "source":         a.get("source", ""),
                "url":            a.get("url", ""),
                "sentiment":      (a.get("sentiment") or {}).get("ensemble_score"),
                "sentiment_label":(a.get("sentiment") or {}).get("ensemble_label"),
                "article_idx":    i,
            })

    # Deduplicate: keep highest-severity event per (date, type)
    best: Dict[str, Dict] = {}
    for ev in raw:
        key = f"{ev['date']}_{ev['type']}"
        if key not in best or ev["severity"] > best[key]["severity"]:
            best[key] = ev

    events = sorted(best.values(), key=lambda e: e["date"], reverse=True)
    return events


def event_summary(events: List[Dict]) -> Dict:
    counts = Counter(e["type"] for e in events)
    summary = []
    for et, n in counts.most_common():
        summary.append({
            "type": et,
            "display": EVENT_TYPES[et]["display"],
            "count": n,
            "color": EVENT_TYPES[et]["color"],
        })
    return {"total": len(events), "by_type": summary}
