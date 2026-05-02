import streamlit as st
from data_collector import collect_data
from indexer import build_index, trag_retrieve
from generator import generate_answer

st.set_page_config(page_title="T-RAG Financial Intelligence", layout="wide")

st.title("📈 Financial Market Intelligence using T-RAG")
st.caption("Temporal-Aware Retrieval-Augmented Generation — IE624 Project")

with st.sidebar:
    st.header("🔧 Settings")
    company_name = st.text_input("Company Name", value="Apple")
    ticker = st.text_input("Stock Ticker", value="AAPL")
    days = st.slider("Days of data to fetch", min_value=7, max_value=30, value=30)
    alpha = st.slider("Alpha (semantic vs time balance)", 0.0, 1.0, 0.7, 0.05)
    top_k = st.slider("Number of articles to retrieve", 3, 10, 5)
    fetch_btn = st.button("🔄 Fetch Data", use_container_width=True)

if "data" not in st.session_state:
    st.session_state.data = None
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if fetch_btn:
    with st.spinner(f"Fetching data for {company_name}..."):
        st.session_state.data = collect_data(company_name, ticker, days)
        st.session_state.index, st.session_state.metadata, st.session_state.embeddings = build_index(
            st.session_state.data["news"]
        )
        st.session_state.company_name = company_name
    st.success(f"✅ Loaded {len(st.session_state.data['news'])} articles and {len(st.session_state.data['stock'])} days of stock data.")

if st.session_state.data:
    st.subheader("📊 Recent Stock Prices")
    stock_df = st.session_state.data["stock"]
    dates = [d["Date"] for d in stock_df]
    prices = [d["Close"] for d in stock_df]
    st.line_chart({"Close Price ($)": prices}, use_container_width=True)

    st.divider()
    st.subheader("💬 Ask a Question")
    query = st.text_input("Your question", placeholder="e.g. How did earnings affect the stock price?")
    ask_btn = st.button("🔍 Get Answer", use_container_width=True)

    if ask_btn and query:
        with st.spinner("Retrieving relevant articles and generating answer..."):
            results = trag_retrieve(
                query,
                st.session_state.index,
                st.session_state.metadata,
                st.session_state.embeddings,
                top_k=top_k,
                alpha=alpha
            )
            answer = generate_answer(
                query,
                results,
                st.session_state.data["stock"],
                st.session_state.company_name
            )

        st.subheader("🤖 Answer")
        st.markdown(answer)

        st.divider()
        st.subheader("📰 Retrieved Articles (T-RAG Ranked)")
        for i, r in enumerate(results, 1):
            with st.expander(f"#{i} | Score: {r['score']} | {r['published_at'][:10]} | {r['source']}"):
                st.markdown(f"**{r['title']}**")
                st.write(r['description'])
else:
    st.info("👈 Enter a company name and ticker in the sidebar, then click **Fetch Data** to begin.")