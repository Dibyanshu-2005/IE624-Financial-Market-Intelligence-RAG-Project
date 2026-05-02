import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def format_articles_as_context(articles):
    """Format retrieved articles into a context string for the LLM."""
    context = ""
    for i, a in enumerate(articles, 1):
        context += f"[Article {i} | Date: {a['published_at'][:10]} | Source: {a['source']}]\n"
        context += f"Title: {a['title']}\n"
        context += f"Summary: {a['description']}\n\n"
    return context

def format_stock_as_context(stock_data):
    """Format stock price data into a readable string."""
    if not stock_data:
        return "No stock data available."
    context = "Recent stock prices (Date | Close Price):\n"
    for entry in stock_data[-10:]:  # Last 10 trading days
        context += f"  {entry['Date']} | ${entry['Close']:.2f}\n"
    return context

def generate_answer(query, retrieved_articles, stock_data, company_name):
    """Send context + query to Groq and get a structured answer."""

    articles_context = format_articles_as_context(retrieved_articles)
    stock_context = format_stock_as_context(stock_data)

    prompt = f"""You are a financial analyst assistant. Answer the user's question using ONLY the provided news articles and stock data. Always cite the article date and source when making a claim.

Company: {company_name}

--- RECENT NEWS ARTICLES ---
{articles_context}

--- STOCK PRICE DATA ---
{stock_context}

--- USER QUESTION ---
{query}

--- YOUR ANSWER ---
Provide a clear, structured answer. Include:
1. A direct answer to the question
2. Key events or news that explain the situation (cite sources and dates)
3. What the stock price data shows
4. A brief sentiment summary (positive / negative / mixed)
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content