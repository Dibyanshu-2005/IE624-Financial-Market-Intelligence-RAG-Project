# Financial Market Intelligence using Temporal-Aware RAG (T-RAG)
**IE624 — Generative and Agentic AI | IIT Bombay 2026**
Authors: Dibyanshu Dash, Devesh Rawat, Gaurav

---

## What This Project Does

This is a **Temporal-Aware Retrieval-Augmented Generation (T-RAG)** system for financial market intelligence. In plain English:

- You type a question like *"How have Apple's earnings affected its stock price?"*
- The system fetches recent news articles and stock price data for that company
- Instead of just finding semantically similar articles (like normal RAG), it also prioritizes **recent** articles using a time-decay formula
- It feeds the retrieved articles to an LLM (Groq/Llama) which generates a structured, citation-backed answer

The core scoring formula (from the report) is:

```
Score(d, q) = α × semantic_similarity(q, d) + (1 - α) × e^(−λ × Δt)
```

Where `α` controls the balance between semantic relevance and temporal recency, and `Δt` is the age of the article in days.

---

## Is the T-RAG Actually Working?

**Yes, the core T-RAG functionality is working correctly.**

The `indexer.py` file implements the time-decay scoring formula exactly as described in the report. You can verify this yourself:

- Run the app, fetch data for any company
- The sidebar has an **Alpha slider** (0.0 to 1.0)
- At `alpha = 1.0` → pure semantic RAG (no time awareness)
- At `alpha = 0.7` → T-RAG (70% semantic + 30% temporal)
- At `alpha = 0.0` → pure recency (only newest articles)

The retrieved articles visibly change depending on alpha, which confirms temporal weighting is active.

What is **not** implemented (from the report):
- Neo4j temporal knowledge graph (too complex for the timeline)
- Multi-hop reasoning (only single-step retrieval is done)
- Apache Kafka for real-time streaming
- RAGAS evaluation framework integration

---

## Project Structure

```
IE624_Project/
├── .env                  # API keys — DO NOT commit this
├── .streamlit/
│   └── config.toml       # Suppresses Streamlit file watcher warnings
├── data_collector.py     # Fetches news (NewsAPI) + stock prices (yfinance)
├── indexer.py            # Embeds articles, builds FAISS index, runs T-RAG retrieval
├── generator.py          # Sends retrieved context to Groq LLM, returns answer
├── app.py                # Streamlit web UI — main entry point
├── evaluate.py           # Evaluation script — BROKEN, see known issues below
└── requirements.txt      # Python dependencies
```

---

## Setup Instructions (for new contributors)

### 1. Clone the repo and create a virtual environment

```bash
git clone <your-repo-url>
cd IE624_Project
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your `.env` file

Create a file called `.env` in the project root with:

```
GROQ_API_KEY=your_groq_api_key_here
NEWS_API_KEY=your_newsapi_key_here
```

- Get a free Groq key at: https://console.groq.com
- Get a free NewsAPI key at: https://newsapi.org

### 4. Fix a yfinance cache bug on macOS

In `data_collector.py`, make sure this line exists right after the imports:

```python
yf.set_tz_cache_location("/tmp/yfinance_cache")
```

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## How to Use the App

1. In the sidebar, enter a **Company Name** (e.g. `Apple`) and **Stock Ticker** (e.g. `AAPL`)
2. Set how many days of data to fetch (7–30)
3. Click **Fetch Data**
4. Once loaded, type a question in the text box (e.g. *"How did earnings affect the stock?"*)
5. Click **Get Answer**

The app will show:
- A line chart of recent stock prices
- A structured LLM-generated answer with citations
- The top retrieved articles ranked by T-RAG score

---

## Known Issues and What Needs Fixing

### 1. `evaluate.py` — BROKEN
The evaluation script does not work correctly. It runs without crashing but the side-by-side comparison output is misaligned and hard to read. What it's supposed to do:
- Compare Regular RAG (`alpha=1.0`) vs T-RAG (`alpha=0.7`) on the same queries
- Show that T-RAG retrieves more recent articles (lower average article age in days)
- Print a clear table showing retrieved articles for both methods

**What needs to be fixed:** The string formatting in the side-by-side print statements is broken on wider titles. Suggested fix: use `pandas` DataFrame to print a proper comparison table, or just print the two result sets sequentially instead of side by side.

### 2. NewsAPI free tier limitation
The free plan only returns articles from the **last 30 days** and caps at 100 requests/day. This limits historical analysis. To fix this properly, integrate a paid news source or scrape financial news from RSS feeds (Yahoo Finance, Reuters, etc.) and store them locally.

### 3. No persistent storage
Every time you fetch data, it re-downloads everything from scratch. A proper implementation would cache the FAISS index and articles to disk so you don't re-fetch on every session. Can be fixed by using `faiss.write_index()` and `pickle` to save/load the index and metadata.

### 4. LLM Model
We are using `llama-3.1-8b-instant` via Groq (free). The original report mentions GPT-4o or LLaMA 3. If you have an OpenAI API key, swap in `gpt-4o` in `generator.py` for better answer quality.

### 5. No knowledge graph
The report proposes a Neo4j-based temporal knowledge graph. This was not implemented. It would require:
- A running Neo4j instance
- Entity extraction from articles (company names, events, people)
- Populating the graph with timestamped relationships
- Querying it during retrieval for multi-hop reasoning

This is the biggest missing piece from the report and would significantly strengthen the project.

---

## Tech Stack

| Component | Tool Used |
|---|---|
| News data | NewsAPI.org |
| Stock data | yfinance |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (IndexFlatIP) |
| LLM | Groq API (llama-3.1-8b-instant) |
| Orchestration | Manual Python pipeline |
| UI | Streamlit |

---

## LLM in generator.py — quick reference

To change the model, open `generator.py` and change this line:

```python
model="llama-3.1-8b-instant",
```

Other free Groq models you can try:
- `llama3-70b-8192` (smarter, slower)
- `mixtral-8x7b-32768` (good for long contexts)