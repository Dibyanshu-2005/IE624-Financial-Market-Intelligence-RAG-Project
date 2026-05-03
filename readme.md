# Financial Market Intelligence using Temporal-Aware RAG (T-RAG)
**IE624 — Generative and Agentic AI | IIT Bombay 2026**
Authors: Dibyanshu Dash, Devesh Rawat, Rushabh

---

## What This Project Does

This system answers questions about a company's financial trajectory by combining four ideas:

1. **Temporal-Aware RAG (T-RAG)** — semantic retrieval blended with a time-decay term so recent, relevant news ranks higher.
2. **Multi-signal sentiment** — every article scored by **FinBERT** (financial-domain BERT), **VADER**, and **Alpha Vantage's pre-computed per-ticker sentiment**, ensembled into a single signal.
3. **Event detection** — articles auto-tagged into a small business taxonomy (earnings, guidance, M&A, regulatory, leadership, analyst-action, partnership, product-launch, stock-action, macro) using Alpha Vantage topics + a keyword/regex layer.
4. **Event-impact analytics** — each detected event linked to next-day and 5-day stock returns + volatility.

You can ask things like *"How has Apple's sentiment changed in the past year?"* or *"Which guidance updates moved the stock most?"* — and you get a structured answer **plus** an interactive timeline, an event log, and an impact chart.

---

## Core Scoring Formula

```
Score(d, q) = α · cosine_similarity(q, d) + (1 − α) · e^(−λ · Δt)
```

- `α` = semantic ↔ temporal balance (sidebar slider; α=1.0 → pure RAG, α=0.7 → T-RAG default)
- `λ` = time-decay rate (sidebar slider)
- `Δt` = age of article in days

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Streamlit UI (5 tabs)                       │
│   Q&A · Sentiment Timeline · Events · Event Impact · Eval      │
└──────────┬─────────────────────────────────────────────────────┘
           │
┌──────────▼──────────┐  ┌──────────────────┐  ┌────────────────┐
│  data_collector.py  │  │    sentiment.py  │  │    events.py   │
│  Alpha Vantage      │  │    FinBERT       │  │  Topic +       │
│  + yfinance         │─▶│    + VADER       │─▶│  keyword       │
│  + NewsAPI fallback │  │    + AV ensemble │  │  tagger        │
└──────────┬──────────┘  └──────────┬───────┘  └────────┬───────┘
           │                        │                   │
           ▼                        ▼                   ▼
┌─────────────────────┐  ┌───────────────────┐  ┌────────────────┐
│     indexer.py      │  │    analytics.py   │  │  generator.py  │
│  FAISS IndexFlatIP  │  │  sentiment×price  │  │  Groq Llama    │
│  + on-disk cache    │  │  event×returns    │  │  3.1-8B        │
│  + T-RAG retrieve   │  │  Pearson corr     │  │  (sentiment-   │
└─────────────────────┘  └───────────────────┘  │   aware prompt)│
                                                └────────────────┘
```

---

## Project Structure

```
IE624-Financial-Market-Intelligence-RAG-Project/
├── .env                  # API keys — DO NOT commit
├── .streamlit/
│   └── config.toml
├── cache/                # on-disk cache for news, prices, FAISS index
├── app.py                # Streamlit UI — 5 tabs
├── data_collector.py     # Alpha Vantage + yfinance + NewsAPI fallback
├── sentiment.py          # FinBERT + VADER + Alpha Vantage ensemble
├── events.py             # Event detection (topics + keyword rules)
├── analytics.py          # Sentiment×price + event-impact correlation
├── indexer.py            # FAISS T-RAG retrieval + persistence
├── generator.py          # Groq LLM (sentiment-aware prompt)
├── evaluate.py           # RAG vs T-RAG metrics (now using pandas)
├── readme.md
└── requirements.txt
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Dibyanshu-2005/IE624-Financial-Market-Intelligence-RAG-Project.git
cd IE624-Financial-Market-Intelligence-RAG-Project
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
# venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download FinBERT (~440 MB). To skip FinBERT, uncheck the sidebar checkbox — the app falls back to VADER + Alpha Vantage ensemble.

### 3. Create your `.env`

```env
GROQ_API_KEY=your_groq_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_newsapi_key_here       # optional fallback
```

- **Groq** (free, required for LLM answers): https://console.groq.com
- **Alpha Vantage** (free, recommended — gives ~2 years of news + per-ticker sentiment): https://www.alphavantage.co/support/#api-key
- **NewsAPI** (free, optional fallback — last 30 days only): https://newsapi.org

### 4. Run

```bash
streamlit run app.py
```

Open http://localhost:8501.

---

## Demo Flow (for the presentation)

1. **Sidebar** → enter `Apple` / `AAPL`, days = 180, click **Fetch & Analyze**.
2. **Tab 1 (Q&A)** → ask *"How has sentiment around earnings shifted?"* → get a sentiment-aware T-RAG answer with cited articles.
3. **Tab 2 (Sentiment Timeline)** → daily mean sentiment as bars + close price overlaid; Pearson correlations between sentiment and same-day / next-day returns.
4. **Tab 3 (Events)** → auto-detected events filterable by type, sentiment-tagged.
5. **Tab 4 (Event Impact)** → mean next-day return per event type + scatter of sentiment vs next-day return.
6. **Tab 5 (Evaluation)** → run T-RAG vs Regular RAG on the same queries; show that T-RAG retrieves measurably newer articles at small semantic cost.

---

## What's Implemented vs the Original Report

| Capability | Status |
|---|---|
| Temporal-RAG with α/λ scoring | ✅ |
| FAISS vector index | ✅ |
| Multi-source news (Alpha Vantage / NewsAPI / yfinance) | ✅ |
| 2-year historical window | ✅ (Alpha Vantage) |
| Sentiment scoring (FinBERT + VADER + AV ensemble) | ✅ |
| Daily sentiment timeline | ✅ |
| Event detection (10 categories) | ✅ |
| Event-impact correlation with stock | ✅ |
| Persistent FAISS cache | ✅ |
| RAG vs T-RAG evaluation metrics | ✅ |
| Sentiment-aware LLM prompt | ✅ |
| Neo4j temporal knowledge graph | ❌ (deferred — out of scope for this milestone) |
| Multi-hop reasoning over the graph | ❌ |
| Apache Kafka real-time streaming | ❌ |
| RAGAS evaluation | ❌ (custom metrics in evaluate.py instead) |

---

## Tech Stack

| Component | Tool |
|---|---|
| News + per-article sentiment (2yr) | Alpha Vantage News & Sentiment API |
| News fallback (30d) | NewsAPI.org |
| Stock prices | yfinance (Yahoo Finance) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector search | FAISS `IndexFlatIP` |
| Sentiment | FinBERT (`ProsusAI/finbert`) + VADER |
| LLM | Groq `llama-3.1-8b-instant` |
| Charts | Plotly |
| UI | Streamlit |

---

## Key Files

- `data_collector.py` — multi-source news + price ingestion with disk cache.
- `sentiment.py` — FinBERT + VADER + Alpha Vantage ensemble.
- `events.py` — event taxonomy + topic/keyword detection.
- `analytics.py` — daily sentiment merge with prices, event-impact windows, Pearson correlation.
- `indexer.py` — FAISS index with persistence and T-RAG combined scoring.
- `generator.py` — Groq LLM prompt that injects retrieved articles, aggregate sentiment, and detected events.
- `evaluate.py` — pandas-based head-to-head metrics: avg article age, recency score, semantic score, Jaccard overlap.
- `app.py` — Streamlit 5-tab UI.

---

## Running the Evaluation Script Standalone

```bash
python evaluate.py
```

Prints a summary metrics DataFrame plus per-query side-by-side comparisons.

---

## Notes

- **Alpha Vantage free tier** = 25 requests/day. The disk cache (`cache/`) makes re-runs free until news changes.
- **FinBERT first run** downloads ~440 MB. Sentiment scoring is slow on CPU; uncheck the FinBERT box to skip it for a faster demo.
- The Neo4j knowledge graph and multi-hop agentic reasoning from the original report are deferred — they'd be the natural next sprint.
