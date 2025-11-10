#!/usr/bin/env python3
"""Test script to verify Varrock schema integration."""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent / 'Alpha-Crucible-Quant' / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv()

# Import from main
from main import (
    get_all_tickers, 
    resolve_ticker_to_company_uid,
    get_main_db_connection,
    get_ore_db_connection
)

def test_varrock_integration():
    """Test Varrock schema integration."""
    main_conn = None
    ore_conn = None
    
    try:
        print("=" * 60)
        print("Testing Varrock Schema Integration")
        print("=" * 60)
        
        # Test main database connection
        print("\n1. Testing main database connection...")
        main_conn = get_main_db_connection()
        print("   ✓ Main database connection successful")
        
        # Test ORE database connection
        print("\n2. Testing ORE database connection...")
        ore_conn = get_ore_db_connection()
        print("   ✓ ORE database connection successful")
        
        # Test ticker query from varrock.tickers
        print("\n3. Testing ticker query from varrock.tickers...")
        tickers = get_all_tickers(main_conn)
        print(f"   ✓ Found {len(tickers)} distinct tickers")
        
        if tickers:
            print(f"\n   Sample tickers (first 10):")
            for i, ticker in enumerate(sorted(tickers)[:10], 1):
                print(f"   {i}. {ticker}")
            if len(tickers) > 10:
                print(f"   ... and {len(tickers) - 10} more")
        
        # Test company_uid resolution
        print("\n4. Testing ticker to company_uid resolution...")
        if tickers:
            test_ticker = sorted(tickers)[0]
            company_uid = resolve_ticker_to_company_uid(main_conn, test_ticker)
            if company_uid:
                print(f"   ✓ Resolved {test_ticker} -> company_uid: {company_uid}")
            else:
                print(f"   ⚠ Could not resolve {test_ticker} to company_uid")
            
            # Test a few more
            resolved_count = 0
            for ticker in list(tickers)[:5]:
                uid = resolve_ticker_to_company_uid(main_conn, ticker)
                if uid:
                    resolved_count += 1
            print(f"   ✓ Resolved {resolved_count}/5 sample tickers to company_uid")
        else:
            print("   ⚠ No tickers found to test resolution")
        
        # Test company_uid column existence
        print("\n5. Testing signal_raw.company_uid column...")
        with main_conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'signal_raw' AND column_name = 'company_uid'
                )
            """)
            has_company_uid = cursor.fetchone()[0]
            if has_company_uid:
                print("   ✓ signal_raw.company_uid column exists")
            else:
                print("   ⚠ signal_raw.company_uid column does not exist (will insert without it)")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if main_conn:
            main_conn.close()
            print("\n✓ Closed main database connection")
        if ore_conn:
            ore_conn.close()
            print("✓ Closed ORE database connection")

if __name__ == "__main__":
    success = test_varrock_integration()
    sys.exit(0 if success else 1)

