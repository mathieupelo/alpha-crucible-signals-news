#!/usr/bin/env python3
"""Test script to debug the processing logic."""

import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import date, timedelta

env_path = Path(__file__).parent.parent / 'Alpha-Crucible-Quant' / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv()

from main import (
    get_all_tickers_with_company_uid,
    get_news_for_ticker_and_date,
    get_sentiment_for_date_range,
    get_main_db_connection,
    get_ore_db_connection,
    is_signal_processed
)

# Test with yesterday (which has news)
test_date = date.today() - timedelta(days=1)
print(f"Testing with date: {test_date}")
print("=" * 60)

# Connect to databases
main_conn = get_main_db_connection()
ore_conn = get_ore_db_connection()

# Get tickers
ticker_map = get_all_tickers_with_company_uid(main_conn)
print(f"\n1. Found {len(ticker_map)} tickers with company_uid")

# Test a few tickers
test_tickers = list(ticker_map.keys())[:5]
print(f"\n2. Testing first 5 tickers: {test_tickers}")

for ticker in test_tickers:
    print(f"\n  {ticker}:")
    company_uid = ticker_map[ticker]
    print(f"    company_uid: {company_uid}")
    
    # Check news
    news = get_news_for_ticker_and_date(ore_conn, ticker, test_date)
    print(f"    News for {test_date}: {len(news)} articles")
    
    # Check sentiment data
    sentiment = get_sentiment_for_date_range(ore_conn, ticker, test_date, days=28)
    print(f"    Sentiment data (last 28 days): {len(sentiment)} items")
    
    # Check if signal already processed
    processed = is_signal_processed(main_conn, ticker, test_date, 'SENTIMENT_YFINANCE_NEWS', company_uid)
    print(f"    Signal already processed: {processed}")

main_conn.close()
ore_conn.close()

