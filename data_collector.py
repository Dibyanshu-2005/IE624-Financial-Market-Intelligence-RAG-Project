"""
Multi-source data collector for the Temporal-RAG Financial Intelligence system.

Sources:
  News   — Alpha Vantage (2yr history + sentiment), NewsAPI (30d), yfinance headlines
  Stock  — yfinance OHLCV (decades of history)
  Filings— SEC EDGAR 8-K / 10-K / 10-Q (US stocks, no key required)
  Social — StockTwits (no key), Reddit via PRAW (needs REDDIT_* env vars)

All sources are combined and deduplicated by title.  News articles from
non-AV sources are filtered to ensure company relevance before adding.

Free-tier limits:
  Alpha Vantage : 25 req/day → aggressively cached (6h TTL)
  NewsAPI       : 100 req/day, 30d window
  SEC EDGAR     : no limit (polite 1 req/s recommended)
  StockTwits    : no auth, ~30 most recent posts per symbol
  Reddit/PRAW   : 60 req/min on free tier
"""

import os
import json
import time
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
yf.set_tz_cache_location("/tmp/yfinance_cache")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY          = os.getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID      = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET  = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT     = os.getenv("REDDIT_USER_AGENT", "FinancialRAG/1.0")

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_HOURS = 6

# SEC EDGAR requires a descriptive User-Agent; update the email if deploying
_SEC_HEADERS = {"User-Agent": "FinancialRAG research-bot contact@example.com"}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(kind: str, key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{kind}_{h}.json"


def _read_cache(path: Path, ttl_hours: int = CACHE_TTL_HOURS):
    if not path.exists():
        return None
    age_h = (time.time() - path.stat().st_mtime) / 3600
    if age_h > ttl_hours:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(path: Path, data) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"[cache] write failed: {e}")


# ---------------------------------------------------------------------------
# Relevance helpers
# ---------------------------------------------------------------------------
def _filter_by_relevance(articles: List[Dict], ticker: str, min_relevance: float) -> List[Dict]:
    """Keep AV articles where the per-ticker relevance score meets the threshold."""
    result = []
    for a in articles:
        rel = a.get("av_ticker_relevance")
        if rel is None or float(rel) >= min_relevance:
            result.append(a)
    return result


def _is_company_relevant(article: Dict, ticker: str, company_name: str) -> bool:
    """Return True if title+description mentions the ticker or company name."""
    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
    if ticker.lower() in text:
        return True
    for word in company_name.lower().split():
        if len(word) > 3 and word in text:
            return True
    return False


# ---------------------------------------------------------------------------
# Stock prices (yfinance)
# ---------------------------------------------------------------------------
def get_stock_data(ticker: str, days: int = 30) -> List[Dict]:
    cache_key = f"{ticker}_{days}"
    cache_file = _cache_path("stock", cache_key)
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    end = datetime.today()
    start = end - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )
    if df.empty:
        return []

    df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Return"] = df["Close"].pct_change().fillna(0).round(4)
    records = df.to_dict(orient="records")

    for r in records:
        for k, v in list(r.items()):
            if hasattr(v, "item"):
                r[k] = v.item()

    _write_cache(cache_file, records)
    return records


# ---------------------------------------------------------------------------
# News — Alpha Vantage
# ---------------------------------------------------------------------------
def _alpha_vantage_news(ticker: str, days: int, min_relevance: float = 0.25) -> Optional[List[Dict]]:
    if not ALPHA_VANTAGE_API_KEY:
        return None

    cache_key = f"{ticker}_{days}_av"
    cache_file = _cache_path("news_av", cache_key)
    cached = _read_cache(cache_file)
    if cached is not None:
        return _filter_by_relevance(cached, ticker, min_relevance)

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    time_from = start.strftime("%Y%m%dT%H%M")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "limit": 1000,
        "sort": "LATEST",
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        payload = resp.json()
    except Exception as e:
        print(f"[Alpha Vantage] request failed: {e}")
        return None

    if "feed" not in payload:
        msg = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
        if msg:
            print(f"[Alpha Vantage] {msg[:160]}")
        return None

    articles = []
    for item in payload.get("feed", []):
        ts_raw = item.get("time_published", "")
        try:
            published_iso = datetime.strptime(ts_raw, "%Y%m%dT%H%M%S").isoformat()
        except Exception:
            published_iso = datetime.utcnow().isoformat()

        ticker_sent = None
        ticker_relevance = None
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                try:
                    ticker_sent = float(ts.get("ticker_sentiment_score", 0))
                    ticker_relevance = float(ts.get("relevance_score", 0))
                except Exception:
                    pass
                break

        try:
            overall_sent = float(item.get("overall_sentiment_score", 0))
        except Exception:
            overall_sent = 0.0

        topics = [t.get("topic") for t in item.get("topics", []) if t.get("topic")]

        articles.append({
            "title": item.get("title", "").strip(),
            "description": item.get("summary", "").strip(),
            "content": item.get("summary", "").strip(),
            "published_at": published_iso,
            "source": item.get("source", "AlphaVantage"),
            "url": item.get("url", ""),
            "topics": topics,
            "data_source": "news",
            "av_overall_sentiment": overall_sent,
            "av_overall_label": item.get("overall_sentiment_label", "Neutral"),
            "av_ticker_sentiment": ticker_sent,
            "av_ticker_relevance": ticker_relevance,
        })

    _write_cache(cache_file, articles)
    return _filter_by_relevance(articles, ticker, min_relevance)


# ---------------------------------------------------------------------------
# News — NewsAPI
# ---------------------------------------------------------------------------
def _newsapi_news(company_name: str, days: int) -> List[Dict]:
    if not NEWS_API_KEY:
        return []

    cache_key = f"{company_name}_{days}_napi"
    cache_file = _cache_path("news_napi", cache_key)
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    days = min(days, 30)
    end = datetime.today()
    start = end - timedelta(days=days)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{company_name}"',
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"[NewsAPI] request failed: {e}")
        return []

    articles = []
    for a in data.get("articles", []):
        if not (a.get("title") and a.get("description")):
            continue
        articles.append({
            "title": a["title"],
            "description": a.get("description", ""),
            "content": a.get("content", ""),
            "published_at": a.get("publishedAt", datetime.utcnow().isoformat()),
            "source": (a.get("source") or {}).get("name", "NewsAPI"),
            "url": a.get("url", ""),
            "topics": [],
            "data_source": "news",
            "av_overall_sentiment": None,
            "av_overall_label": None,
            "av_ticker_sentiment": None,
            "av_ticker_relevance": None,
        })

    _write_cache(cache_file, articles)
    return articles


# ---------------------------------------------------------------------------
# News — yfinance headlines
# ---------------------------------------------------------------------------
def _yfinance_news(ticker: str) -> List[Dict]:
    cache_key = f"{ticker}_yf"
    cache_file = _cache_path("news_yf", cache_key)
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"[yfinance news] failed: {e}")
        return []

    articles = []
    for it in items:
        content = it.get("content", it)
        title = content.get("title") or it.get("title", "")
        summary = content.get("summary") or it.get("summary", "") or title
        pub_date = content.get("pubDate") or it.get("providerPublishTime")
        if isinstance(pub_date, (int, float)):
            published_iso = datetime.utcfromtimestamp(pub_date).isoformat()
        elif isinstance(pub_date, str):
            published_iso = pub_date.replace("Z", "")
        else:
            published_iso = datetime.utcnow().isoformat()
        publisher = (
            (content.get("provider", {}) or {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else it.get("publisher", "Yahoo")
        )
        if not title:
            continue
        url = (
            (content.get("canonicalUrl", {}) or {}).get("url", "")
            if isinstance(content.get("canonicalUrl"), dict)
            else it.get("link", "")
        )
        articles.append({
            "title": title,
            "description": summary,
            "content": summary,
            "published_at": published_iso,
            "source": publisher or "Yahoo Finance",
            "url": url,
            "topics": [],
            "data_source": "news",
            "av_overall_sentiment": None,
            "av_overall_label": None,
            "av_ticker_sentiment": None,
            "av_ticker_relevance": None,
        })

    _write_cache(cache_file, articles)
    return articles


# ---------------------------------------------------------------------------
# SEC EDGAR filings (no API key required)
# ---------------------------------------------------------------------------
def _sec_get_cik(ticker: str) -> Optional[str]:
    """Map a ticker symbol to an SEC CIK number."""
    cache_file = CACHE_DIR / "sec_tickers.json"
    data = _read_cache(cache_file, ttl_hours=24 * 7)

    if data is None:
        try:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=_SEC_HEADERS,
                timeout=30,
            )
            data = resp.json()
            _write_cache(cache_file, data)
        except Exception as e:
            print(f"[SEC EDGAR] CIK lookup failed: {e}")
            return None

    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"])
    return None


def _sec_edgar_filings(ticker: str, days: int = 365) -> List[Dict]:
    """Fetch recent 8-K, 10-K, 10-Q filings from SEC EDGAR as article dicts."""
    cache_key = f"{ticker}_{days}_sec"
    cache_file = _cache_path("sec", cache_key)
    cached = _read_cache(cache_file, ttl_hours=12)
    if cached is not None:
        return cached

    cik = _sec_get_cik(ticker)
    if not cik:
        print(f"[SEC EDGAR] CIK not found for {ticker} — skipping filings")
        return []

    try:
        url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"[SEC EDGAR] submissions fetch failed: {e}")
        return []

    filings = data.get("filings", {}).get("recent", {})
    if not filings:
        return []

    forms        = filings.get("form", [])
    dates        = filings.get("filingDate", [])
    descriptions = filings.get("primaryDocDescription") or [""] * len(forms)
    accessions   = filings.get("accessionNumber", [])

    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    target_forms = {"8-K", "10-K", "10-Q"}
    company_name_sec = data.get("name", ticker)

    type_labels = {
        "8-K":  "Current Report (8-K)",
        "10-K": "Annual Report (10-K)",
        "10-Q": "Quarterly Report (10-Q)",
    }
    # Map filing types to event topics for downstream event detection
    filing_topics = {
        "8-K":  ["Financial Markets"],
        "10-K": ["Earnings"],
        "10-Q": ["Earnings"],
    }

    articles = []
    for i, (form, date, acc) in enumerate(zip(forms, dates, accessions)):
        if form not in target_forms:
            continue
        if date < cutoff:
            continue

        desc = descriptions[i] if i < len(descriptions) else ""
        acc_clean = acc.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{acc}.htm"
        )
        label = type_labels.get(form, form)
        articles.append({
            "title": f"{company_name_sec} — {label} ({date})",
            "description": (
                f"{label} filed with the SEC on {date} by {company_name_sec}. "
                + (f"Primary document: {desc}." if desc else "")
            ),
            "content": f"{label} filing dated {date}.",
            "published_at": f"{date}T09:00:00",
            "source": "SEC EDGAR",
            "url": filing_url,
            "topics": filing_topics.get(form, []),
            "data_source": "filing",
            "form_type": form,
            "av_overall_sentiment": None,
            "av_overall_label": None,
            "av_ticker_sentiment": None,
            "av_ticker_relevance": 1.0,
        })

    _write_cache(cache_file, articles)
    print(f"  [SEC EDGAR] {len(articles)} filings for {ticker}")
    return articles


# ---------------------------------------------------------------------------
# Social — StockTwits (no API key)
# ---------------------------------------------------------------------------
def _stocktwits_posts(ticker: str) -> List[Dict]:
    """Fetch the most recent StockTwits stream for a ticker (~30 posts)."""
    cache_key = f"{ticker}_st"
    cache_file = _cache_path("social_st", cache_key)
    cached = _read_cache(cache_file, ttl_hours=2)
    if cached is not None:
        return cached

    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        resp = requests.get(url, timeout=20)
        data = resp.json()
    except Exception as e:
        print(f"[StockTwits] request failed: {e}")
        return []

    if data.get("response", {}).get("status") != 200:
        msg = data.get("response", {}).get("message", "unknown error")
        print(f"[StockTwits] API error: {msg}")
        return []

    articles = []
    for msg in data.get("messages", []):
        body = (msg.get("body") or "").strip()
        if len(body) < 15:
            continue

        created = msg.get("created_at", "")
        try:
            published_iso = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").isoformat()
        except Exception:
            published_iso = datetime.utcnow().isoformat()

        sentiment_info = (msg.get("entities") or {}).get("sentiment")
        av_score = None
        sentiment_label = None
        if sentiment_info and sentiment_info.get("basic"):
            sentiment_label = sentiment_info["basic"]  # "Bullish" / "Bearish"
            av_score = 0.6 if sentiment_label == "Bullish" else -0.6

        username = (msg.get("user") or {}).get("username", "anonymous")

        articles.append({
            "title": f"${ticker} — {body[:100]}{'…' if len(body) > 100 else ''}",
            "description": body,
            "content": body,
            "published_at": published_iso,
            "source": f"StockTwits/@{username}",
            "url": f"https://stocktwits.com/{username}/message/{msg.get('id', '')}",
            "topics": [],
            "data_source": "social",
            "av_overall_sentiment": av_score,
            "av_overall_label": sentiment_label,
            "av_ticker_sentiment": av_score,
            "av_ticker_relevance": 0.9,
        })

    _write_cache(cache_file, articles)
    print(f"  [StockTwits] {len(articles)} posts for {ticker}")
    return articles


# ---------------------------------------------------------------------------
# Social — Reddit via PRAW (optional: needs REDDIT_CLIENT_ID + SECRET)
# ---------------------------------------------------------------------------
def _reddit_posts(ticker: str, company_name: str, days: int = 30) -> List[Dict]:
    """Fetch Reddit posts from financial subreddits. Requires PRAW credentials."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return []

    cache_key = f"{ticker}_{min(days, 30)}_reddit"
    cache_file = _cache_path("social_reddit", cache_key)
    cached = _read_cache(cache_file, ttl_hours=4)
    if cached is not None:
        return cached

    try:
        import praw
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
    except ImportError:
        print("[Reddit] praw not installed — run: pip install praw")
        return []
    except Exception as e:
        print(f"[Reddit] PRAW init failed: {e}")
        return []

    cutoff = datetime.utcnow() - timedelta(days=min(days, 30))
    search_query = f"${ticker} OR \"{company_name}\""
    subreddits = ["stocks", "investing", "wallstreetbets", "StockMarket"]

    articles = []
    seen: set = set()

    for sub_name in subreddits:
        try:
            for post in reddit.subreddit(sub_name).search(
                search_query, time_filter="month", limit=25, sort="relevance"
            ):
                if post.id in seen:
                    continue
                seen.add(post.id)

                pub_time = datetime.utcfromtimestamp(post.created_utc)
                if pub_time < cutoff:
                    continue
                if not _is_company_relevant(
                    {"title": post.title, "description": post.selftext or ""},
                    ticker,
                    company_name,
                ):
                    continue

                body = (post.selftext or "").strip()[:500]
                description = body if body else post.title

                # Upvote ratio [0,1] → signed sentiment proxy [-1,+1]
                av_score = round((post.upvote_ratio - 0.5) * 2, 3)

                articles.append({
                    "title": post.title.strip(),
                    "description": description,
                    "content": description,
                    "published_at": pub_time.isoformat(),
                    "source": f"Reddit/r/{sub_name}",
                    "url": f"https://reddit.com{post.permalink}",
                    "topics": [],
                    "data_source": "social",
                    "av_overall_sentiment": av_score,
                    "av_overall_label": (
                        "Bullish" if av_score > 0.1 else
                        "Bearish" if av_score < -0.1 else "Neutral"
                    ),
                    "av_ticker_sentiment": av_score,
                    "av_ticker_relevance": 0.8,
                })
        except Exception as e:
            print(f"[Reddit] r/{sub_name} failed: {e}")

    _write_cache(cache_file, articles)
    print(f"  [Reddit] {len(articles)} posts for {ticker}")
    return articles


# ---------------------------------------------------------------------------
# Combined news fetch
# ---------------------------------------------------------------------------
def get_news_articles(company_name: str, ticker: str, days: int = 30) -> List[Dict]:
    """Combine all news sources, deduplicated by title."""
    articles: List[Dict] = []
    seen_titles: set = set()

    def _add(new_list: List[Dict]) -> None:
        for a in new_list:
            key = a.get("title", "").strip().lower()[:80]
            if key and key not in seen_titles:
                seen_titles.add(key)
                articles.append(a)

    # 1) Alpha Vantage — best quality, long history, per-ticker relevance scored
    av = _alpha_vantage_news(ticker, days)
    if av:
        _add(av)
        print(f"  [Alpha Vantage] {len(av)} articles")

    # 2) NewsAPI — filter by company mention before adding
    napi = _newsapi_news(company_name, days)
    if napi:
        relevant = [a for a in napi if _is_company_relevant(a, ticker, company_name)]
        _add(relevant)
        print(f"  [NewsAPI] {len(relevant)}/{len(napi)} relevant articles added")

    # 3) yfinance headlines
    yfn = _yfinance_news(ticker)
    if yfn:
        _add(yfn)
        print(f"  [yfinance] {len(yfn)} headlines added")

    if not articles:
        print("  [warn] no news returned — set ALPHA_VANTAGE_API_KEY or NEWS_API_KEY in .env")

    return sorted(articles, key=lambda a: a.get("published_at", ""), reverse=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def collect_data(company_name: str, ticker: str, days: int = 30) -> Dict:
    """Fetch news, stock prices, SEC filings, and social media for a company.

    Returns:
        {
          "news"   : list of news article dicts,
          "stock"  : list of OHLCV dicts,
          "reports": list of SEC filing dicts,
          "social" : list of social media post dicts,
        }
    """
    print(f"\nFetching news for {company_name} ({ticker}, last {days}d)...")
    news = get_news_articles(company_name, ticker, days)
    print(f"  Total news articles: {len(news)}")

    print(f"Fetching stock data for {ticker}...")
    stock = get_stock_data(ticker, days)
    print(f"  Got {len(stock)} trading days")

    print(f"Fetching SEC filings for {ticker}...")
    reports = _sec_edgar_filings(ticker, days)

    print(f"Fetching social media for {ticker}...")
    social: List[Dict] = []
    social += _stocktwits_posts(ticker)
    social += _reddit_posts(ticker, company_name, days)
    print(f"  Total social posts: {len(social)}")

    return {
        "news":    news,
        "stock":   stock,
        "reports": reports,
        "social":  social,
    }
