#!/usr/bin/env python3
"""Check unique constraints on signal_raw table."""

import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / 'Alpha-Crucible-Quant' / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv()

from main import get_main_db_connection

conn = get_main_db_connection()
cursor = conn.cursor()

# Check unique constraints
cursor.execute("""
    SELECT conname, pg_get_constraintdef(oid) 
    FROM pg_constraint 
    WHERE conrelid = 'signal_raw'::regclass 
    AND contype = 'u';
""")

print("Unique constraints on signal_raw:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

# Check what columns exist
cursor.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'signal_raw'
    ORDER BY ordinal_position;
""")

print("\nColumns in signal_raw:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} (nullable: {row[2]})")

conn.close()

