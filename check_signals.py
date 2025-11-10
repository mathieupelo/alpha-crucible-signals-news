#!/usr/bin/env python3
"""Check what signals are in the database for a specific date."""

import os
import sys
import psycopg2
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / 'Alpha-Crucible-Quant' / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv()

def get_main_db_connection():
    """Get connection to main database."""
    try:
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return psycopg2.connect(database_url)
        
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT', '5432')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        database = os.getenv('DB_NAME')
        
        if not all([host, user, password, database]):
            raise ValueError("Database connection parameters not found")
        
        return psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def check_date(conn, check_date: date):
    """Check signals for a specific date."""
    print(f"\n{'='*60}")
    print(f"Checking signals for date: {check_date}")
    print(f"{'='*60}")
    
    # Get all tickers in universe
    with conn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT ticker FROM universe_tickers ORDER BY ticker;")
        all_tickers = {row[0] for row in cursor.fetchall()}
        print(f"Total tickers in universe: {len(all_tickers)}")
    
    # Get signals for this date
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT ticker, value, metadata
            FROM signal_raw
            WHERE signal_name = 'SENTIMENT_YFINANCE_NEWS'
            AND asof_date = %s
            ORDER BY ticker;
        """, (check_date,))
        
        rows = cursor.fetchall()
        signals = {row[0]: {'value': row[1], 'metadata': row[2]} for row in rows}
        print(f"Signals found in database: {len(signals)}")
    
    # Find missing tickers
    tickers_with_signals = set(signals.keys())
    missing_tickers = all_tickers - tickers_with_signals
    
    print(f"\nTickers WITH signals ({len(tickers_with_signals)}):")
    for ticker in sorted(tickers_with_signals):
        value = signals[ticker]['value']
        if value is None:
            print(f"  {ticker}: NULL")
        else:
            print(f"  {ticker}: {value:.6f}")
    
    if missing_tickers:
        print(f"\nTickers MISSING signals ({len(missing_tickers)}):")
        for ticker in sorted(missing_tickers)[:20]:  # Show first 20
            print(f"  {ticker}: (no signal)")
        if len(missing_tickers) > 20:
            print(f"  ... and {len(missing_tickers) - 20} more")
    else:
        print(f"\n✓ All tickers have signals for this date")

def main():
    """Main function."""
    conn = None
    try:
        conn = get_main_db_connection()
        check_date(conn, date(2025, 10, 26))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()


