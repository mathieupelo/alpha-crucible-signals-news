#!/usr/bin/env python3
"""Debug summary script to show what the main script would do."""

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
    is_signal_processed,
    is_sentiment_processed
)

# Test with today
test_date = date.today()
print("=" * 60)
print(f"DEBUG SUMMARY for {test_date}")
print("=" * 60)

# Connect to databases
main_conn = get_main_db_connection()
ore_conn = get_ore_db_connection()

# Get tickers
ticker_map = get_all_tickers_with_company_uid(main_conn)
print(f"\n1. TICKERS: Found {len(ticker_map)} tickers with company_uid")

# Step 2 Analysis
print(f"\n2. STEP 2 - SENTIMENT PROCESSING for {test_date}:")
print("-" * 60)
tickers_with_news = 0
total_news = 0
total_unprocessed_sentiment = 0

for ticker in list(ticker_map.keys())[:10]:  # Check first 10
    news = get_news_for_ticker_and_date(ore_conn, ticker, test_date)
    if news:
        tickers_with_news += 1
        total_news += len(news)
        unprocessed = sum(1 for n in news if not is_sentiment_processed(ore_conn, n['id']))
        total_unprocessed_sentiment += unprocessed
        if unprocessed > 0:
            print(f"  {ticker}: {len(news)} news, {unprocessed} unprocessed sentiments")

print(f"\n  Summary (first 10 tickers):")
print(f"    Tickers with news: {tickers_with_news}")
print(f"    Total news articles: {total_news}")
print(f"    Unprocessed sentiments: {total_unprocessed_sentiment}")

# Step 3 Analysis
print(f"\n3. STEP 3 - SIGNAL AGGREGATION for {test_date}:")
print("-" * 60)
tickers_with_sentiment = 0
tickers_already_processed = 0
tickers_no_sentiment = 0

for ticker in list(ticker_map.keys())[:10]:  # Check first 10
    company_uid = ticker_map[ticker]
    sentiment = get_sentiment_for_date_range(ore_conn, ticker, test_date, days=28)
    processed = is_signal_processed(main_conn, ticker, test_date, 'SENTIMENT_YFINANCE_NEWS', company_uid)
    
    if processed:
        tickers_already_processed += 1
    elif sentiment:
        tickers_with_sentiment += 1
        print(f"  {ticker}: {len(sentiment)} sentiment items, would aggregate")
    else:
        tickers_no_sentiment += 1
        print(f"  {ticker}: No sentiment data, would insert NULL")

print(f"\n  Summary (first 10 tickers):")
print(f"    Already processed: {tickers_already_processed}")
print(f"    Would aggregate: {tickers_with_sentiment}")
print(f"    Would insert NULL: {tickers_no_sentiment}")

# Check all tickers
print(f"\n4. FULL ANALYSIS (all {len(ticker_map)} tickers):")
print("-" * 60)
all_already_processed = 0
all_would_aggregate = 0
all_would_null = 0

for ticker, company_uid in ticker_map.items():
    processed = is_signal_processed(main_conn, ticker, test_date, 'SENTIMENT_YFINANCE_NEWS', company_uid)
    if processed:
        all_already_processed += 1
    else:
        sentiment = get_sentiment_for_date_range(ore_conn, ticker, test_date, days=28)
        if sentiment:
            all_would_aggregate += 1
        else:
            all_would_null += 1

print(f"  Already processed: {all_already_processed}")
print(f"  Would aggregate: {all_would_aggregate}")
print(f"  Would insert NULL: {all_would_null}")

main_conn.close()
ore_conn.close()

print("\n" + "=" * 60)
print("DEBUG SUMMARY COMPLETE")
print("=" * 60)

