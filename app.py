"""
Financial Market Intelligence using Temporal-Aware RAG (T-RAG)

Tabs:
  1. Q&A            — sentiment-aware T-RAG answers over news + filings + social
  2. Sentiment      — daily sentiment timeline overlaid on stock price
                      with major-event markers (severity ≥ 0.7) + toggle
  3. Events         — auto-detected business events table
  4. Event Impact   — events vs next-day / 5-day stock returns
  5. Evaluation     — Regular RAG vs T-RAG side-by-side metrics
"""

import os
import pandas as pd
import streamlit as st

from data_collector import collect_data
from indexer import build_index, trag_retrieve
from generator import generate_answer
from sentiment import score_articles, aggregate_daily
from events import detect_events, event_summary, EVENT_TYPES
from analytics import (
    merge_sentiment_with_prices,
    event_impact,
    sentiment_price_correlation,
    aggregate_event_type_impact,
)
from evaluate import compare_rag_vs_trag

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

st.set_page_config(page_title="T-RAG Financial Intelligence", layout="wide", page_icon="📈")
st.title("📈 Financial Market Intelligence using T-RAG")
st.caption(
    "Temporal-Aware RAG · Sentiment Timelines · Event Detection · "
    "Event-Impact Correlation · SEC Filings · Social Sentiment — IE624 Project"
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔧 Settings")
    company_name = st.text_input("Company Name", value="Apple")
    ticker       = st.text_input("Stock Ticker",  value="AAPL")
    days = st.slider(
        "Days of historical data", 30, 730, 180, step=30,
        help="Alpha Vantage gives up to ~2 years. NewsAPI is capped at 30 days.",
    )
    alpha = st.slider("Alpha (semantic ↔ time balance)", 0.0, 1.0, 0.7, 0.05)
    lam   = st.slider(
        "λ (time-decay rate)", 0.005, 0.2, 0.05, 0.005,
        help="Higher λ = recent articles weighted more aggressively.",
    )
    top_k       = st.slider("Top-K articles to retrieve", 3, 15, 5)
    use_finbert = st.checkbox(
        "Use FinBERT (slower, ~440 MB)", value=False,
        help="Financial-domain BERT for better sentiment. Disable for faster loading.",
    )
    fetch_btn = st.button("🔄 Fetch & Analyze", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("**API keys (`.env`):**")
    for name, ok in [
        ("Alpha Vantage — news+sentiment, 2yr", bool(os.getenv("ALPHA_VANTAGE_API_KEY"))),
        ("Groq — LLM inference",                bool(os.getenv("GROQ_API_KEY"))),
        ("NewsAPI — fallback, 30d",             bool(os.getenv("NEWS_API_KEY"))),
        ("Reddit — social posts (optional)",    bool(os.getenv("REDDIT_CLIENT_ID"))),
    ]:
        st.markdown(f"- {'✅' if ok else '❌'} {name}")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for k in ["data", "index", "metadata", "embeddings", "events",
          "daily_sentiment", "company_name", "ticker"]:
    if k not in st.session_state:
        st.session_state[k] = None


# ---------------------------------------------------------------------------
# Fetch & analyze pipeline
# ---------------------------------------------------------------------------
if fetch_btn:
    with st.spinner(f"Fetching news + prices + filings + social for {company_name} ({ticker})…"):
        st.session_state.data = collect_data(company_name, ticker, days)

    news    = st.session_state.data.get("news",    [])
    reports = st.session_state.data.get("reports", [])
    social  = st.session_state.data.get("social",  [])
    stock   = st.session_state.data.get("stock",   [])

    if not news and not reports and not social:
        st.error(
            "⚠️ No documents returned. Set `ALPHA_VANTAGE_API_KEY` in `.env` "
            "for 2-year history, or `NEWS_API_KEY` for a 30-day fallback."
        )
    else:
        # Score sentiment on news + filings; social already carries AV sentiment labels
        all_text_docs = news + reports + social
        with st.spinner("Scoring sentiment with FinBERT + VADER…"):
            scored_all = score_articles(all_text_docs, use_finbert=use_finbert)

        n_news    = len(news)
        n_reports = len(reports)
        scored_news_and_reports = scored_all[: n_news + n_reports]

        with st.spinner("Detecting events…"):
            st.session_state.events = detect_events(scored_news_and_reports)

        with st.spinner("Building T-RAG index (news + filings + social)…"):
            idx, meta, emb = build_index(scored_all)
            st.session_state.index     = idx
            st.session_state.metadata  = meta
            st.session_state.embeddings = emb

        st.session_state.daily_sentiment = aggregate_daily(scored_news_and_reports)
        st.session_state.company_name    = company_name
        st.session_state.ticker          = ticker

        # Persist scored data back into session for Tab 2 social display
        st.session_state.data["news"]    = scored_all[:n_news]
        st.session_state.data["reports"] = scored_all[n_news: n_news + n_reports]
        st.session_state.data["social"]  = scored_all[n_news + n_reports:]

        st.success(
            f"✅ {len(news)} news · {len(reports)} SEC filings · {len(social)} social posts | "
            f"{len(st.session_state.events)} events | {len(stock)} trading days"
        )


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
if not st.session_state.data:
    st.info("👈 Pick a company in the sidebar and click **Fetch & Analyze** to begin.")
    st.markdown("""
### What this app does
- **Temporal-RAG retrieval** — semantic similarity + time-decay so recent documents rank higher when relevant.
- **Multi-source ingestion** — financial news (Alpha Vantage, NewsAPI), SEC filings (8-K/10-K/10-Q via EDGAR), social sentiment (StockTwits + Reddit).
- **Sentiment analytics** — FinBERT (financial-domain BERT) + VADER + Alpha Vantage pre-computed scores, ensembled.
- **Event detection** — articles auto-tagged: earnings, guidance, M&A, regulatory, leadership, analyst-action, partnership, product-launch, stock-action, macro.
- **Event-impact markers** — major events (severity ≥ 0.7) shown as markers on the sentiment vs price chart.
- **Event-impact correlation** — each event linked to next-day and 5-day stock returns.
- **Evaluation** — Regular RAG (α=1.0) vs T-RAG head-to-head metrics.

**Required `.env` keys:** `ALPHA_VANTAGE_API_KEY`, `GROQ_API_KEY`
**Optional:** `NEWS_API_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
""")
    st.stop()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_qa, tab_sent, tab_events, tab_impact, tab_eval = st.tabs([
    "💬 Q&A", "📊 Sentiment Timeline", "🗓️ Events", "📈 Event Impact", "🧪 Evaluation",
])


# ─── Tab 1: Q&A ─────────────────────────────────────────────────────────────
with tab_qa:
    st.subheader(f"Ask about {st.session_state.company_name}")

    # Show data-source summary
    n_news_s    = len(st.session_state.data.get("news", []))
    n_reports_s = len(st.session_state.data.get("reports", []))
    n_social_s  = len(st.session_state.data.get("social", []))
    st.caption(
        f"Index contains **{n_news_s}** news articles · "
        f"**{n_reports_s}** SEC filings · "
        f"**{n_social_s}** social posts"
    )

    query = st.text_input(
        "Your question",
        placeholder="e.g. How has sentiment changed around the last earnings report?",
        key="qa_query",
    )
    ask = st.button("🔍 Get Answer", use_container_width=True, key="ask_btn")

    if ask and query:
        with st.spinner("Retrieving and generating answer…"):
            results = trag_retrieve(
                query,
                st.session_state.index,
                st.session_state.metadata,
                st.session_state.embeddings,
                top_k=top_k, alpha=alpha, lam=lam,
            )
            answer = generate_answer(
                query, results,
                st.session_state.data["stock"],
                st.session_state.company_name,
                events=st.session_state.events,
            )

        st.markdown("### 🤖 Answer")
        st.markdown(answer)
        st.divider()
        st.markdown("### 📰 Retrieved Documents (T-RAG ranked)")

        for i, r in enumerate(results, 1):
            sent     = r.get("sentiment") or {}
            src_type = r.get("data_source", "news")
            type_icon = {"filing": "📄", "social": "💬", "news": "📰"}.get(src_type, "📰")
            badge = ""
            if sent.get("ensemble_label"):
                emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
                    sent["ensemble_label"], "⚪"
                )
                badge = f"{emoji} {sent['ensemble_label']} ({sent.get('ensemble_score', 0):+.2f}) · "
            with st.expander(
                f"{type_icon} #{i} | T-RAG: {r['score']} "
                f"(sem {r['semantic_score']}, time {r['temporal_score']}) | "
                f"{badge}{r['published_at'][:10]} | {r['source']}"
            ):
                st.markdown(f"**{r['title']}**")
                st.write(r["description"])
                if r.get("url"):
                    st.markdown(f"[Read source →]({r['url']})")


# ─── Tab 2: Sentiment Timeline ───────────────────────────────────────────────
with tab_sent:
    st.subheader("Sentiment timeline vs stock price")

    daily = st.session_state.daily_sentiment or []
    stock = st.session_state.data["stock"]
    events_all = st.session_state.events or []

    if not daily:
        st.warning("No sentiment data — sentiment scoring may not have run.")
    else:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        avg_sent = sum(d["mean_sentiment"] for d in daily) / len(daily)
        pos = sum(d["pos_n"] for d in daily)
        neg = sum(d["neg_n"] for d in daily)
        neu = sum(d["neu_n"] for d in daily)
        c1.metric("Mean sentiment", f"{avg_sent:+.3f}")
        c2.metric("Positive articles", pos)
        c3.metric("Negative articles", neg)
        c4.metric("Neutral articles",  neu)

        # Social sentiment KPIs (StockTwits + Reddit)
        social_posts = st.session_state.data.get("social", [])
        if social_posts:
            bull = sum(
                1 for p in social_posts
                if (p.get("av_overall_label") or "").lower() in ("bullish", "positive")
            )
            bear = sum(
                1 for p in social_posts
                if (p.get("av_overall_label") or "").lower() in ("bearish", "negative")
            )
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Social posts", len(social_posts))
            sc2.metric("Bullish 🐂", bull)
            sc3.metric("Bearish 🐻", bear)

        # Event markers toggle
        show_event_markers = st.checkbox(
            "📌 Show major event markers (severity ≥ 0.7) on chart",
            value=True,
            key="show_events_toggle",
        )

        if HAVE_PLOTLY:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Sentiment bars
            sent_dates = [d["date"] for d in daily]
            sent_vals  = [d["mean_sentiment"] for d in daily]
            fig.add_trace(
                go.Bar(
                    x=sent_dates, y=sent_vals,
                    name="Daily mean sentiment",
                    marker_color=["#2ca02c" if v > 0 else "#d62728" for v in sent_vals],
                    opacity=0.55,
                ),
                secondary_y=False,
            )

            # Stock price line
            if stock:
                fig.add_trace(
                    go.Scatter(
                        x=[s["Date"] for s in stock],
                        y=[s["Close"] for s in stock],
                        name="Close price", mode="lines",
                        line=dict(color="#1f77b4", width=2),
                    ),
                    secondary_y=True,
                )

            # Major event markers on the price line
            if show_event_markers and events_all:
                high_sev = [e for e in events_all if e["severity"] >= 0.7]

                # Deduplicate: one marker per (date, type)
                seen_ev: set = set()
                unique_events = []
                for ev in sorted(high_sev, key=lambda x: -x["severity"]):
                    key = f"{ev['date']}_{ev['type']}"
                    if key not in seen_ev:
                        seen_ev.add(key)
                        unique_events.append(ev)
                        if len(unique_events) >= 10:
                            break

                if unique_events and stock:
                    price_by_date = {s["Date"]: s["Close"] for s in stock}
                    sorted_dates  = sorted(price_by_date.keys())

                    ev_x, ev_y, ev_text, ev_color, ev_size = [], [], [], [], []
                    for ev in unique_events:
                        # Find nearest trading day on or after event date
                        candidates = [d for d in sorted_dates if d >= ev["date"]]
                        trade_date = candidates[0] if candidates else sorted_dates[-1]
                        price = price_by_date.get(trade_date, 0)

                        ev_x.append(ev["date"])
                        ev_y.append(price)
                        ev_text.append(
                            f"<b>{ev['type_display']}</b><br>"
                            f"{ev['date']} | sev={ev['severity']:.2f}<br>"
                            f"{ev['headline'][:70]}"
                        )
                        ev_color.append(EVENT_TYPES[ev["type"]]["color"])
                        ev_size.append(8 + ev["severity"] * 14)

                    fig.add_trace(
                        go.Scatter(
                            x=ev_x, y=ev_y,
                            mode="markers",
                            name="Major Events",
                            marker=dict(
                                size=ev_size,
                                color=ev_color,
                                symbol="diamond",
                                line=dict(width=1.5, color="white"),
                            ),
                            text=ev_text,
                            hoverinfo="text+x",
                        ),
                        secondary_y=True,
                    )

            fig.update_layout(
                height=460,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", y=-0.2),
                hovermode="x unified",
            )
            fig.update_yaxes(
                title_text="Sentiment (−1 to +1)", secondary_y=False, range=[-1, 1]
            )
            fig.update_yaxes(title_text="Close Price ($)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pd.DataFrame(daily).set_index("date")["mean_sentiment"])

        # Pearson correlation metrics
        corr = sentiment_price_correlation(daily, stock)
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Pearson corr (sentiment vs same-day return)",
            "n/a" if corr["same_day"] is None else f"{corr['same_day']:+.3f}",
            help="Correlation computed on actual news dates mapped to nearest trading day.",
        )
        c2.metric(
            "Pearson corr (sentiment vs next-day return)",
            "n/a" if corr["next_day"] is None else f"{corr['next_day']:+.3f}",
        )
        c3.metric(
            "News dates used for correlation",
            corr["n"],
            help="Number of distinct news dates with both sentiment and a matching trading day.",
        )

        if corr["same_day"] is None and corr["n"] < 3:
            st.info(
                f"ℹ️ Only **{corr['n']}** news dates overlap with trading days — need ≥ 3 to compute Pearson. "
                "Try increasing the **Days of historical data** slider or ensuring Alpha Vantage returns sufficient articles."
            )

        st.markdown("### Daily sentiment table")
        st.dataframe(pd.DataFrame(daily), use_container_width=True, hide_index=True)

        # Social media posts table
        if social_posts:
            with st.expander(f"💬 Social media posts ({len(social_posts)} total)"):
                social_df = pd.DataFrame([{
                    "date":      p.get("published_at", "")[:10],
                    "source":    p.get("source", ""),
                    "sentiment": (p.get("av_overall_label") or "").capitalize(),
                    "text":      p.get("description", "")[:120],
                    "url":       p.get("url", ""),
                } for p in sorted(social_posts, key=lambda x: x.get("published_at", ""), reverse=True)])
                st.dataframe(
                    social_df, use_container_width=True, hide_index=True,
                    column_config={"url": st.column_config.LinkColumn("link", display_text="open")},
                )


# ─── Tab 3: Events ───────────────────────────────────────────────────────────
with tab_events:
    st.subheader("Auto-detected business events")
    events = st.session_state.events or []

    if not events:
        st.warning("No events detected.")
    else:
        summary = event_summary(events)
        st.caption(
            f"Total events: **{summary['total']}** across "
            f"{len(summary['by_type'])} categories."
        )

        if HAVE_PLOTLY and summary["by_type"]:
            sb = summary["by_type"]
            fig = go.Figure(go.Bar(
                x=[s["display"] for s in sb],
                y=[s["count"]   for s in sb],
                marker_color=[s["color"] for s in sb],
                text=[s["count"] for s in sb],
                textposition="outside",
            ))
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="# articles tagged", xaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

        c1, _ = st.columns([1, 3])
        with c1:
            type_filter = st.multiselect(
                "Filter by type",
                options=sorted(EVENT_TYPES.keys()),
                default=[],
                format_func=lambda x: EVENT_TYPES[x]["display"],
            )

        # SEC filings breakdown
        reports = st.session_state.data.get("reports", [])
        if reports:
            n_8k  = sum(1 for r in reports if r.get("form_type") == "8-K")
            n_10k = sum(1 for r in reports if r.get("form_type") == "10-K")
            n_10q = sum(1 for r in reports if r.get("form_type") == "10-Q")
            st.caption(
                f"📄 SEC Filings in dataset: **{len(reports)}** total — "
                f"{n_8k} × 8-K · {n_10k} × 10-K · {n_10q} × 10-Q"
            )

        filtered = [e for e in events if (not type_filter or e["type"] in type_filter)]

        df = pd.DataFrame([{
            "date":     e["date"],
            "type":     e["type_display"],
            "severity": e["severity"],
            "sentiment":e.get("sentiment"),
            "headline": e["headline"][:120],
            "source":   e["source"],
            "url":      e.get("url", ""),
        } for e in filtered])

        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "url":      st.column_config.LinkColumn("link", display_text="open"),
                "sentiment":st.column_config.NumberColumn("sentiment", format="%+.2f"),
                "severity": st.column_config.ProgressColumn("severity", min_value=0, max_value=1),
            },
        )


# ─── Tab 4: Event Impact ─────────────────────────────────────────────────────
with tab_impact:
    st.subheader("Event-impact analysis")
    events = st.session_state.events or []
    stock  = st.session_state.data["stock"]

    if not events or not stock:
        st.warning("Need both events and stock data.")
    else:
        with st.spinner("Computing impact windows…"):
            impacted = event_impact(events, stock)
            agg      = aggregate_event_type_impact(impacted)

        st.markdown("#### Mean next-day return by event type")
        if HAVE_PLOTLY and agg:
            colors = [EVENT_TYPES[r["type"]]["color"] for r in agg]
            fig = go.Figure(go.Bar(
                x=[f"{r['type_display']} (n={r['count']})" for r in agg],
                y=[r["mean_next_day_return"] * 100 for r in agg],
                marker_color=colors,
                text=[f"{r['mean_next_day_return']*100:+.2f}%" for r in agg],
                textposition="outside",
            ))
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="Mean next-day return (%)",
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
            )
            st.plotly_chart(fig, use_container_width=True)
        elif agg:
            st.dataframe(pd.DataFrame(agg), use_container_width=True, hide_index=True)

        st.markdown("#### Each event: sentiment vs next-day return")
        scatter_rows = [
            r for r in impacted
            if r.get("return_next_day") is not None and r.get("sentiment") is not None
        ]
        if HAVE_PLOTLY and scatter_rows:
            fig = go.Figure(go.Scatter(
                x=[r["sentiment"] for r in scatter_rows],
                y=[r["return_next_day"] * 100 for r in scatter_rows],
                mode="markers",
                marker=dict(
                    size=[max(6, r["severity"] * 16) for r in scatter_rows],
                    color=[EVENT_TYPES[r["type"]]["color"] for r in scatter_rows],
                    line=dict(width=0.5, color="#444"),
                ),
                text=[
                    f"{r['date']} · {r['type_display']}<br>{r['headline'][:90]}"
                    for r in scatter_rows
                ],
                hoverinfo="text+x+y",
            ))
            fig.update_layout(
                height=380, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Article sentiment (−1..+1)",
                yaxis_title="Next-day return (%)",
                xaxis=dict(zeroline=True),
                yaxis=dict(zeroline=True),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### All events with impact")
        df = pd.DataFrame([{
            "date":         e["date"],
            "type":         e["type_display"],
            "severity":     e["severity"],
            "sentiment":    e.get("sentiment"),
            "next_day_ret": e.get("return_next_day"),
            "5d_return":    e.get("return_5d"),
            "5d_vol":       e.get("volatility_5d"),
            "headline":     e["headline"][:90],
        } for e in impacted])
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "sentiment":    st.column_config.NumberColumn(format="%+.2f"),
                "next_day_ret": st.column_config.NumberColumn(format="%+.4f"),
                "5d_return":    st.column_config.NumberColumn(format="%+.4f"),
                "5d_vol":       st.column_config.NumberColumn(format="%.4f"),
                "severity":     st.column_config.ProgressColumn(min_value=0, max_value=1),
            },
        )


# ─── Tab 5: Evaluation ───────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Regular RAG vs T-RAG — head-to-head")
    st.caption("Same queries, same documents — α=1.0 (pure semantic) vs α as set in sidebar.")

    default_queries = (
        "How did the most recent earnings affect the stock price?\n"
        "What are analysts saying about future growth?\n"
        "Any recent news about product launches or partnerships?\n"
        "How has investor sentiment changed recently?"
    )
    queries_text = st.text_area("Queries (one per line)", value=default_queries, height=120)
    run_eval     = st.button("🧪 Run evaluation", use_container_width=True)

    if run_eval:
        queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
        with st.spinner("Running both retrievers…"):
            try:
                summary, details = compare_rag_vs_trag(
                    company_name=st.session_state.company_name,
                    ticker=st.session_state.ticker,
                    queries=queries,
                    days=days,
                    top_k=top_k,
                    alpha_trag=alpha,
                    lam=lam,
                )
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                summary, details = None, []

        if summary is not None:
            st.markdown("#### Summary metrics")
            st.dataframe(
                summary, use_container_width=True, hide_index=True,
                column_config={
                    "Δ_age_days": st.column_config.NumberColumn(
                        "Δ age (RAG − T-RAG)", format="%.2f",
                        help="Positive = T-RAG retrieved newer articles.",
                    ),
                },
            )
            for q, d in details:
                with st.expander(f"📋 Per-rank comparison — {q}"):
                    st.dataframe(d, use_container_width=True, hide_index=True)
            st.info(
                "**Reading the table:** higher Δ_age_days and TRAG_recency mean T-RAG is "
                "successfully shifting toward more recent documents. RAG_semantic vs "
                "TRAG_semantic shows the similarity cost of recency weighting. "
                "RAG_src / TRAG_src shows document type (news / filing / social)."
            )
