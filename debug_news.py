#!/usr/bin/env python3
"""Debug script to check news in ORE database."""

import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import date, timedelta

env_path = Path(__file__).parent.parent / 'Alpha-Crucible-Quant' / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv()

from main import get_ore_db_connection, get_news_for_date

# Connect to ORE
ore_conn = get_ore_db_connection()
cursor = ore_conn.cursor()

# Check total news count
cursor.execute("SELECT COUNT(*) FROM copper.yfinance_news")
total_news = cursor.fetchone()[0]
print(f"Total news articles in database: {total_news}")

# Check distinct tickers in news
cursor.execute("SELECT DISTINCT ticker FROM copper.yfinance_news ORDER BY ticker")
tickers_in_news = [row[0] for row in cursor.fetchall()]
print(f"\nDistinct tickers in news table ({len(tickers_in_news)}):")
for ticker in tickers_in_news[:20]:
    print(f"  {ticker}")
if len(tickers_in_news) > 20:
    print(f"  ... and {len(tickers_in_news) - 20} more")

# Check date range
cursor.execute("SELECT MIN(DATE(published_date)), MAX(DATE(published_date)) FROM copper.yfinance_news")
date_range = cursor.fetchone()
print(f"\nDate range in news table: {date_range[0]} to {date_range[1]}")

# Check news for yesterday
yesterday = date.today() - timedelta(days=1)
print(f"\nChecking news for {yesterday}:")
news_list = get_news_for_date(ore_conn, yesterday)
print(f"Found {len(news_list)} news articles for {yesterday}")

if news_list:
    print("\nSample news articles:")
    for news in news_list[:5]:
        print(f"  {news['ticker']}: {news['title'][:50]}...")

# Check news for today
today = date.today()
print(f"\nChecking news for {today}:")
news_list_today = get_news_for_date(ore_conn, today)
print(f"Found {len(news_list_today)} news articles for {today}")

# Check recent news (last 7 days)
cursor.execute("""
    SELECT DATE(published_date) as pub_date, COUNT(*) as count
    FROM copper.yfinance_news
    WHERE DATE(published_date) >= %s
    GROUP BY DATE(published_date)
    ORDER BY pub_date DESC
    LIMIT 7
""", (date.today() - timedelta(days=7),))
print("\nNews count by date (last 7 days):")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} articles")

ore_conn.close()

