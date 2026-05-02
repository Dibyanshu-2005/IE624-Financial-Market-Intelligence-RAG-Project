import os
import yfinance as yf
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

yf.set_tz_cache_location("/tmp/yfinance_cache")

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_stock_data(ticker, days=30):
    """Fetch stock price data for a ticker over the past `days` days."""
    stock = yf.Ticker(ticker)
    end = datetime.today()
    start = end - timedelta(days=days)
    df = stock.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
    df = df[['Close', 'Volume']].reset_index()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df.to_dict(orient='records')

def get_news_articles(company_name, days=30):
    """Fetch news articles about a company from NewsAPI."""
    end = datetime.today()
    start = end - timedelta(days=days)

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_name,
        "from": start.strftime('%Y-%m-%d'),
        "to": end.strftime('%Y-%m-%d'),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = []
    for article in data.get("articles", []):
        if article.get("title") and article.get("description"):
            articles.append({
                "title": article["title"],
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "published_at": article["publishedAt"],
                "source": article["source"]["name"]
            })

    return articles

def collect_data(company_name, ticker, days=30):
    """Collect both news and stock data for a company."""
    print(f"Fetching news for {company_name}...")
    news = get_news_articles(company_name, days)
    print(f"  Found {len(news)} articles")

    print(f"Fetching stock data for {ticker}...")
    stock = get_stock_data(ticker, days)
    print(f"  Found {len(stock)} trading days of data")

    return {"news": news, "stock": stock}