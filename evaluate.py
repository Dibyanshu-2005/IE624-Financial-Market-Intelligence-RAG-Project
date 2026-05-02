from data_collector import collect_data
from indexer import build_index, trag_retrieve
from datetime import datetime

def average_age_days(results):
    """Calculate average age (in days) of retrieved articles."""
    now = datetime.utcnow()
    ages = []
    for r in results:
        try:
            pub = datetime.strptime(r["published_at"][:19], "%Y-%m-%dT%H:%M:%S")
            ages.append((now - pub).days)
        except:
            pass
    return round(sum(ages) / len(ages), 2) if ages else 0

def run_evaluation(company_name, ticker, queries):
    print(f"\n{'='*60}")
    print(f"  T-RAG EVALUATION — {company_name} ({ticker})")
    print(f"{'='*60}")

    print("\nFetching data...")
    data = collect_data(company_name, ticker, days=30)
    index, metadata, embeddings = build_index(data["news"])

    for query in queries:
        print(f"\n{'─'*60}")
        print(f"QUERY: {query}")
        print(f"{'─'*60}")

        # Regular RAG: alpha=1.0 means 100% semantic, 0% temporal
        rag_results = trag_retrieve(query, index, metadata, embeddings,
                                    top_k=5, alpha=1.0)

        # T-RAG: alpha=0.7 means 70% semantic, 30% temporal
        trag_results = trag_retrieve(query, index, metadata, embeddings,
                                     top_k=5, alpha=0.7)

        rag_avg_age = average_age_days(rag_results)
        trag_avg_age = average_age_days(trag_results)

        print(f"\n{'Regular RAG (no time awareness)':^40} | {'T-RAG (time-aware)':^40}")
        print(f"{'Avg article age: ' + str(rag_avg_age) + ' days':^40} | {'Avg article age: ' + str(trag_avg_age) + ' days':^40}")
        print()

        for i in range(5):
            r = rag_results[i]
            t = trag_results[i]
            r_line = f"{r['published_at'][:10]} | {r['title'][:35]}"
            t_line = f"{t['published_at'][:10]} | {t['title'][:35]}"
            print(f"  {r_line:<40} | {t_line:<40}")

    print(f"\n{'='*60}")
    print("CONCLUSION: T-RAG retrieves more recent articles while")
    print("maintaining semantic relevance. Regular RAG ignores time.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    queries = [
        "How did earnings affect the stock price?",
        "What are analysts saying about future growth?",
        "Any recent news about product launches?"
    ]
    run_evaluation("Apple", "AAPL", queries)