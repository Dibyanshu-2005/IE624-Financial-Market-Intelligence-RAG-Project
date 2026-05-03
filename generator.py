"""
LLM answer generation using Groq.

Sentiment-aware: retrieved articles carry ensemble sentiment scores so the
prompt can ground sentiment claims in real numbers.  SEC filings and social
media posts are included in context when retrieved via T-RAG.
"""

import os
from typing import List, Dict, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def format_articles_as_context(articles: List[Dict]) -> str:
    ctx = ""
    for i, a in enumerate(articles, 1):
        sent = a.get("sentiment") or {}
        sent_str = ""
        if sent.get("ensemble_score") is not None:
            sent_str = f" | Sentiment: {sent.get('ensemble_label','?')} ({sent.get('ensemble_score'):+.2f})"
        src_type = a.get("data_source", "news")
        type_tag = {"filing": "📄 SEC Filing", "social": "💬 Social", "news": "📰 News"}.get(src_type, "📰 News")
        ctx += f"[{type_tag} #{i} | {a['published_at'][:10]} | {a['source']}{sent_str}]\n"
        ctx += f"Title: {a['title']}\n"
        ctx += f"Summary: {a['description']}\n\n"
    return ctx


def format_stock_as_context(stock_data: List[Dict]) -> str:
    if not stock_data:
        return "No stock data available."
    ctx = "Recent stock prices (Date | Close | 1d Return):\n"
    for entry in stock_data[-15:]:
        ret = entry.get("Return", 0) or 0
        ctx += f"  {entry['Date']} | ${entry['Close']:.2f} | {ret:+.2%}\n"
    return ctx


def format_sentiment_summary(articles: List[Dict]) -> str:
    scores = [
        (a.get("sentiment") or {}).get("ensemble_score")
        for a in articles
        if (a.get("sentiment") or {}).get("ensemble_score") is not None
    ]
    if not scores:
        return ""
    pos = sum(1 for s in scores if s >= 0.15)
    neg = sum(1 for s in scores if s <= -0.15)
    neu = len(scores) - pos - neg
    avg = sum(scores) / len(scores)
    return (
        f"\n--- AGGREGATE SENTIMENT (retrieved articles) ---\n"
        f"  Positive: {pos}  Neutral: {neu}  Negative: {neg}\n"
        f"  Mean score: {avg:+.3f}\n"
    )


def generate_answer(
    query: str,
    retrieved_articles: List[Dict],
    stock_data: List[Dict],
    company_name: str,
    events: Optional[List[Dict]] = None,
) -> str:
    articles_ctx = format_articles_as_context(retrieved_articles)
    stock_ctx    = format_stock_as_context(stock_data)
    sent_ctx     = format_sentiment_summary(retrieved_articles)

    events_ctx = ""
    if events:
        events_ctx = "\n--- DETECTED EVENTS (high severity first) ---\n"
        for ev in sorted(events, key=lambda e: -e.get("severity", 0))[:10]:
            events_ctx += (
                f"  {ev['date']} | {ev['type_display']} | "
                f"sev={ev['severity']:.2f} | {ev['headline'][:80]}\n"
            )

    prompt = f"""You are a financial analyst assistant. Answer using ONLY the data provided below. Cite article dates and sources for every claim.

Company: {company_name}

--- RETRIEVED DOCUMENTS (T-RAG ranked: news, filings, social posts) ---
{articles_ctx}{sent_ctx}{events_ctx}
--- STOCK PRICE DATA ---
{stock_ctx}

--- USER QUESTION ---
{query}

--- YOUR ANSWER ---
Provide a structured answer covering:
1. Direct answer to the question.
2. Key events / news driving the situation (cite dates and sources).
3. What stock price movement and sentiment scores show.
4. Brief sentiment summary grounded in the aggregate score above.
Keep to 4-6 short paragraphs. If SEC filings are referenced, highlight the filing type and date."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content
