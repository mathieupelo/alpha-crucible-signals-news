#!/usr/bin/env python3
"""
Alpha Crucible Signals - News
Fetches news data from ORE database, calculates sentiment signals, and stores in main database.
"""

import os
import sys
import logging
import json
import psycopg2
from psycopg2.extras import execute_values, Json
from psycopg2 import sql
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple, Set
from pathlib import Path
from dotenv import load_dotenv
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Lazy import transformers to avoid breaking if not installed
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables - try Alpha-Crucible-Quant directory first
env_path = Path(__file__).parent.parent / 'Alpha-Crucible-Quant' / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
    logger.info(f"Loaded .env from: {env_path}")
else:
    # Try immediate parent directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=True)
        logger.info(f"Loaded .env from: {env_path}")
    else:
        # Fallback to current directory
        load_dotenv()
        logger.info("Loaded .env from current directory")

# FinBERT model configuration
FINBERT_MODEL = "yiyanghkust/finbert-tone"


class FinBERTSentimentAnalyzer:
    """FinBERT sentiment analyzer for financial news."""
    
    def __init__(self):
        """Initialize FinBERT model."""
        self.tokenizer = None
        self.model = None
        self.device = None
        self._model_loaded = False
        
    def _load_model(self):
        """Lazy load FinBERT model."""
        if self._model_loaded:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for sentiment analysis")
        
        try:
            logger.info("Loading FinBERT model for sentiment analysis...")
            self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Use GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            self._model_loaded = True
            logger.info(f"FinBERT model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using FinBERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment label, score (-1 to 1), and probabilities
        """
        if not self._model_loaded:
            self._load_model()
        
        try:
            # Combine title and summary if needed
            if not text or not text.strip():
                return {
                    "label": "neutral",
                    "score": 0.0,
                    "positive_prob": 0.33,
                    "negative_prob": 0.33,
                    "neutral_prob": 0.34,
                    "confidence": 0.0
                }
            
            # Tokenize and encode (limit to 512 tokens)
            inputs = self.tokenizer(
                text[:512],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: 0=positive, 1=negative, 2=neutral
            scores = predictions[0].cpu().numpy()
            positive_prob = float(scores[0])
            negative_prob = float(scores[1])
            neutral_prob = float(scores[2])
            
            # Convert to -1 to 1 scale: (positive_prob - negative_prob)
            sentiment_score = positive_prob - negative_prob
            
            # Determine label
            label_idx = scores.argmax()
            labels = ["positive", "negative", "neutral"]
            label = labels[label_idx]
            confidence = float(scores[label_idx])
            
            return {
                "label": label,
                "score": sentiment_score,
                "positive_prob": positive_prob,
                "negative_prob": negative_prob,
                "neutral_prob": neutral_prob,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}", exc_info=True)
            # Return neutral on error
            return {
                "label": "neutral",
                "score": 0.0,
                "positive_prob": 0.33,
                "negative_prob": 0.33,
                "neutral_prob": 0.34,
                "confidence": 0.0,
                "error": str(e)
            }


def get_ore_db_connection():
    """Get connection to ORE database (for fetching news)."""
    try:
        # Try ORE_DATABASE_URL first
        database_url = os.getenv('ORE_DATABASE_URL')
        if not database_url:
            # Try alternative variable name (used in Airflow)
            database_url = os.getenv('DATABASE_ORE_URL')
        
        if database_url:
            logger.info("Connecting to ORE database using database URL")
            return psycopg2.connect(database_url)
        
        # Fall back to individual connection parameters
        host = os.getenv('ORE_DB_HOST')
        port = os.getenv('ORE_DB_PORT', '5432')
        user = os.getenv('ORE_DB_USER')
        password = os.getenv('ORE_DB_PASSWORD')
        database = os.getenv('ORE_DB_NAME')
        
        if not all([host, user, password, database]):
            raise ValueError(
                "ORE database connection requires either ORE_DATABASE_URL, DATABASE_ORE_URL, or "
                "(ORE_DB_HOST, ORE_DB_USER, ORE_DB_PASSWORD, ORE_DB_NAME)"
            )
        
        logger.info(f"Connecting to ORE database at {host}:{port}")
        return psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
    except Exception as e:
        logger.error(f"Error connecting to ORE database: {e}")
        raise


def get_main_db_connection():
    """Get connection to main database (for storing signals)."""
    try:
        # Try DATABASE_URL first
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            logger.info("Connecting to main database using DATABASE_URL")
            return psycopg2.connect(database_url)
        
        # Fall back to individual connection parameters
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT', '5432')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        database = os.getenv('DB_NAME')
        
        if not all([host, user, password, database]):
            raise ValueError(
                "Main database connection requires either DATABASE_URL or "
                "(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)"
            )
        
        logger.info(f"Connecting to main database at {host}:{port}")
        return psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
    except Exception as e:
        logger.error(f"Error connecting to main database: {e}")
        raise


def create_sentiment_table(ore_conn):
    """Create the copper.yfinance_news_sentiment table if it doesn't exist."""
    try:
        with ore_conn.cursor() as cursor:
            # Create schema if it doesn't exist
            cursor.execute("CREATE SCHEMA IF NOT EXISTS copper;")
            
            # Create table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS copper.yfinance_news_sentiment (
                    id SERIAL PRIMARY KEY,
                    yfinance_news_id INTEGER NOT NULL,
                    ticker VARCHAR(20) NOT NULL,
                    sentiment_score FLOAT NOT NULL,
                    sentiment_label VARCHAR(20) NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(yfinance_news_id)
                );
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_yfinance_news_sentiment_ticker_date 
                ON copper.yfinance_news_sentiment(ticker, created_at);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_yfinance_news_sentiment_news_id 
                ON copper.yfinance_news_sentiment(yfinance_news_id);
            """)
            
            ore_conn.commit()
            logger.info("Table copper.yfinance_news_sentiment created/verified successfully")
            
    except Exception as e:
        logger.error(f"Error creating sentiment table: {e}")
        ore_conn.rollback()
        raise


def get_news_for_date(ore_conn, target_date: date) -> List[Dict[str, Any]]:
    """Get all news articles for a specific date from ORE database."""
    try:
        with ore_conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, ticker, title, summary, publisher, link, published_date, image_url
                FROM copper.yfinance_news
                WHERE DATE(published_date) = %s
                ORDER BY ticker, published_date;
            """, (target_date,))
            
            rows = cursor.fetchall()
            news_list = []
            for row in rows:
                news_list.append({
                    'id': row[0],
                    'ticker': row[1],
                    'title': row[2] or '',
                    'summary': row[3] or '',
                    'publisher': row[4] or '',
                    'link': row[5] or '',
                    'published_date': row[6],
                    'image_url': row[7] or ''
                })
            
            logger.info(f"Found {len(news_list)} news articles for {target_date}")
            return news_list
            
    except Exception as e:
        logger.error(f"Error fetching news for {target_date}: {e}")
        raise


def is_sentiment_processed(ore_conn, news_id: int) -> bool:
    """Check if sentiment has already been calculated for a news article."""
    try:
        with ore_conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM copper.yfinance_news_sentiment 
                WHERE yfinance_news_id = %s;
            """, (news_id,))
            count = cursor.fetchone()[0]
            return count > 0
    except Exception as e:
        logger.error(f"Error checking if sentiment processed for news_id {news_id}: {e}")
        return False


def insert_sentiment(ore_conn, news_id: int, ticker: str, sentiment_data: Dict[str, Any]):
    """Insert or update sentiment for a news article."""
    try:
        with ore_conn.cursor() as cursor:
            # Prepare metadata
            metadata = {
                'positive_prob': sentiment_data['positive_prob'],
                'negative_prob': sentiment_data['negative_prob'],
                'neutral_prob': sentiment_data['neutral_prob'],
                'confidence': sentiment_data['confidence']
            }
            if 'error' in sentiment_data:
                metadata['error'] = sentiment_data['error']
            
            # Use ON CONFLICT DO UPDATE to replace if exists
            cursor.execute("""
                INSERT INTO copper.yfinance_news_sentiment 
                (yfinance_news_id, ticker, sentiment_score, sentiment_label, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (yfinance_news_id) 
                DO UPDATE SET
                    ticker = EXCLUDED.ticker,
                    sentiment_score = EXCLUDED.sentiment_score,
                    sentiment_label = EXCLUDED.sentiment_label,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP;
            """, (
                news_id,
                ticker,
                sentiment_data['score'],
                sentiment_data['label'],
                Json(metadata)
            ))
            
            ore_conn.commit()
            
    except Exception as e:
        logger.error(f"Error inserting sentiment for news_id {news_id}: {e}")
        ore_conn.rollback()
        raise


def get_or_create_signal_id(main_conn, signal_name: str) -> int:
    """Get or create signal record and return its ID."""
    try:
        with main_conn.cursor() as cursor:
            # Check if signals table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'signals'
                )
            """)
            signals_table_exists = cursor.fetchone()[0]
            
            if not signals_table_exists:
                # If signals table doesn't exist, signal_id is not required
                # Return None to indicate we should use signal_name only
                return None
            
            # Try to get existing signal
            cursor.execute("""
                SELECT id FROM signals WHERE name = %s;
            """, (signal_name,))
            result = cursor.fetchone()
            
            if result:
                signal_id = result[0]
                logger.debug(f"Found existing signal_id {signal_id} for {signal_name}")
                return signal_id
            
            # Create new signal record
            cursor.execute("""
                INSERT INTO signals (name, description, created_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id;
            """, (signal_name, f"Sentiment signal from yfinance news"))
            
            signal_id = cursor.fetchone()[0]
            main_conn.commit()
            logger.info(f"Created signal record with id {signal_id} for {signal_name}")
            return signal_id
            
    except Exception as e:
        logger.warning(f"Error getting/creating signal_id for {signal_name}: {e}. Will try without signal_id.")
        main_conn.rollback()
        return None


def is_signal_processed(main_conn, ticker: str, target_date: date, signal_name: str) -> bool:
    """Check if signal has already been calculated for a ticker/date."""
    try:
        with main_conn.cursor() as cursor:
            # Check if signal_id column exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'signal_raw' AND column_name = 'signal_id'
                )
            """)
            has_signal_id = cursor.fetchone()[0]
            
            if has_signal_id:
                # Use signal_id if available
                signal_id = get_or_create_signal_id(main_conn, signal_name)
                if signal_id:
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM signal_raw 
                        WHERE ticker = %s 
                        AND asof_date = %s 
                        AND signal_id = %s;
                    """, (ticker, target_date, signal_id))
                else:
                    # Fallback to signal_name
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM signal_raw 
                        WHERE ticker = %s 
                        AND asof_date = %s 
                        AND signal_name = %s;
                    """, (ticker, target_date, signal_name))
            else:
                # Use signal_name only
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM signal_raw 
                    WHERE ticker = %s 
                    AND asof_date = %s 
                    AND signal_name = %s;
                """, (ticker, target_date, signal_name))
            
            count = cursor.fetchone()[0]
            return count > 0
    except Exception as e:
        logger.error(f"Error checking if signal processed for {ticker}/{target_date}: {e}")
        return False


def get_all_tickers(main_conn) -> Set[str]:
    """Get all distinct tickers from universe_tickers table."""
    try:
        with main_conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT ticker FROM universe_tickers ORDER BY ticker;")
            tickers = {row[0] for row in cursor.fetchall()}
            logger.debug(f"Found {len(tickers)} distinct tickers in database")
            return tickers
    except Exception as e:
        logger.error(f"Error fetching distinct tickers: {e}")
        raise


def get_sentiment_for_date_range(ore_conn, ticker: str, end_date: date, days: int = 7) -> List[Dict[str, Any]]:
    """Get sentiment scores for a ticker over the last N days."""
    try:
        start_date = end_date - timedelta(days=days-1)
        
        with ore_conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    s.yfinance_news_id,
                    s.ticker,
                    s.sentiment_score,
                    s.sentiment_label,
                    s.metadata,
                    n.published_date
                FROM copper.yfinance_news_sentiment s
                JOIN copper.yfinance_news n ON s.yfinance_news_id = n.id
                WHERE s.ticker = %s
                AND DATE(n.published_date) >= %s
                AND DATE(n.published_date) <= %s
                ORDER BY n.published_date DESC;
            """, (ticker, start_date, end_date))
            
            rows = cursor.fetchall()
            sentiment_list = []
            for row in rows:
                sentiment_list.append({
                    'news_id': row[0],
                    'ticker': row[1],
                    'sentiment_score': row[2],
                    'sentiment_label': row[3],
                    'metadata': row[4],
                    'published_date': row[5]
                })
            
            return sentiment_list
            
    except Exception as e:
        logger.error(f"Error fetching sentiment for {ticker} from {start_date} to {end_date}: {e}")
        return []


def calculate_aggregated_sentiment(sentiment_list: List[Dict[str, Any]], target_date: date) -> Dict[str, Any]:
    """
    Calculate aggregated sentiment with 7-day decay weighting.
    
    Decay: Day 0 (most recent) = 1.0, Day 1 = 0.5, Day 2 = 0.25, etc.
    """
    if not sentiment_list:
        return {
            'aggregated_score': 0.0,
            'count': 0,
            'metadata': {}
        }
    
    total_weighted_score = 0.0
    total_weight = 0.0
    sentiment_by_day = {}
    
    for sentiment in sentiment_list:
        pub_date = sentiment['published_date']
        if isinstance(pub_date, datetime):
            pub_date = pub_date.date()
        elif isinstance(pub_date, str):
            pub_date = datetime.fromisoformat(pub_date).date()
        
        # Calculate days ago from target_date
        days_ago = (target_date - pub_date).days
        
        if days_ago < 0 or days_ago >= 7:
            continue  # Skip if outside 7-day window
        
        # Calculate decay weight: 0.5^days_ago
        weight = 0.5 ** days_ago
        
        score = sentiment['sentiment_score']
        total_weighted_score += score * weight
        total_weight += weight
        
        # Track sentiment by day for metadata
        if days_ago not in sentiment_by_day:
            sentiment_by_day[days_ago] = {'count': 0, 'total_score': 0.0, 'total_weight': 0.0}
        sentiment_by_day[days_ago]['count'] += 1
        sentiment_by_day[days_ago]['total_score'] += score
        sentiment_by_day[days_ago]['total_weight'] += weight
    
    if total_weight == 0:
        aggregated_score = 0.0
    else:
        aggregated_score = total_weighted_score / total_weight
    
    # Prepare metadata
    total_news_in_window = sum(day_data['count'] for day_data in sentiment_by_day.values())
    metadata = {
        'news_count': len(sentiment_list),
        'weighted_news_count': total_news_in_window,
        'total_weight': total_weight,
        'sentiment_by_day': sentiment_by_day,
        'calculation_method': '7_day_decay_weighted_average',
        'decay_factor': 0.5
    }
    
    return {
        'aggregated_score': aggregated_score,
        'count': len(sentiment_list),
        'metadata': metadata
    }


def insert_signal(main_conn, ticker: str, target_date: date, signal_name: str, value: Optional[float], metadata: Optional[Dict[str, Any]]):
    """Insert or update signal in signal_raw table. Value can be None if no sentiment data available."""
    try:
        with main_conn.cursor() as cursor:
            # Check if signal_id column exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'signal_raw' AND column_name = 'signal_id'
                )
            """)
            has_signal_id = cursor.fetchone()[0]
            
            # Prepare metadata (can be None)
            metadata_json = Json(metadata) if metadata else None
            
            if has_signal_id:
                # Get or create signal_id
                signal_id = get_or_create_signal_id(main_conn, signal_name)
                if signal_id:
                    # Use signal_id in insert
                    cursor.execute("""
                        INSERT INTO signal_raw 
                        (asof_date, ticker, signal_name, signal_id, value, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (asof_date, ticker, signal_name) 
                        DO UPDATE SET
                            signal_id = EXCLUDED.signal_id,
                            value = EXCLUDED.value,
                            metadata = EXCLUDED.metadata,
                            created_at = CURRENT_TIMESTAMP;
                    """, (
                        target_date,
                        ticker,
                        signal_name,
                        signal_id,
                        value,  # Can be None
                        metadata_json
                    ))
                else:
                    # Fallback: try without signal_id (might fail if NOT NULL)
                    cursor.execute("""
                        INSERT INTO signal_raw 
                        (asof_date, ticker, signal_name, value, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (asof_date, ticker, signal_name) 
                        DO UPDATE SET
                            value = EXCLUDED.value,
                            metadata = EXCLUDED.metadata,
                            created_at = CURRENT_TIMESTAMP;
                    """, (
                        target_date,
                        ticker,
                        signal_name,
                        value,  # Can be None
                        metadata_json
                    ))
            else:
                # Use signal_name only (old schema)
                cursor.execute("""
                    INSERT INTO signal_raw 
                    (asof_date, ticker, signal_name, value, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (asof_date, ticker, signal_name) 
                    DO UPDATE SET
                        value = EXCLUDED.value,
                        metadata = EXCLUDED.metadata,
                        created_at = CURRENT_TIMESTAMP;
                """, (
                    target_date,
                    ticker,
                    signal_name,
                    value,  # Can be None
                    metadata_json
                ))
            
            main_conn.commit()
            
    except Exception as e:
        logger.error(f"Error inserting signal for {ticker}/{target_date}: {e}")
        main_conn.rollback()
        raise


def ensure_value_column_nullable(main_conn):
    """Ensure the value column in signal_raw allows NULL values."""
    try:
        with main_conn.cursor() as cursor:
            # Check if value column is nullable
            cursor.execute("""
                SELECT is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'signal_raw' 
                AND column_name = 'value';
            """)
            result = cursor.fetchone()
            
            if result and result[0] == 'NO':
                # Column is NOT NULL, alter it to allow NULL
                logger.info("Altering signal_raw.value column to allow NULL values...")
                cursor.execute("""
                    ALTER TABLE signal_raw 
                    ALTER COLUMN value DROP NOT NULL;
                """)
                main_conn.commit()
                logger.info("✓ signal_raw.value column now allows NULL values")
            elif result and result[0] == 'YES':
                logger.debug("signal_raw.value column already allows NULL values")
            else:
                logger.warning("Could not determine nullability of signal_raw.value column")
                
    except Exception as e:
        logger.warning(f"Could not alter signal_raw.value column: {e}. Will attempt insert anyway.")
        main_conn.rollback()


def process_date_range(start_date: date, end_date: date, ore_conn, main_conn, analyzer: FinBERTSentimentAnalyzer):
    """Process sentiment calculation for all dates in the range."""
    current_date = start_date
    total_sentiments_processed = 0
    total_signals_processed = 0
    total_errors = 0
    
    while current_date <= end_date:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing date: {current_date}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Get all news for this date
            news_list = get_news_for_date(ore_conn, current_date)
            
            # Get all tickers from main database (needed for NULL signal insertion)
            all_tickers = get_all_tickers(main_conn)
            
            if not news_list:
                logger.info(f"No news found for {current_date}. Inserting NULL signals for all tickers.")
                # Insert NULL signals for all tickers when no news
                for ticker in all_tickers:
                    try:
                        # Check if signal already processed
                        if is_signal_processed(main_conn, ticker, current_date, 'SENTIMENT_YFINANCE_NEWS'):
                            continue
                        
                        # Insert NULL signal
                        insert_signal(
                            main_conn,
                            ticker,
                            current_date,
                            'SENTIMENT_YFINANCE_NEWS',
                            None,  # NULL value
                            {'reason': 'no_news_data_for_date', 'date': str(current_date)}
                        )
                        total_signals_processed += 1
                    except Exception as e:
                        total_errors += 1
                        logger.error(f"ERROR: Failed to insert NULL signal for {ticker} on {current_date}: {e}", exc_info=True)
                        continue
                
                current_date += timedelta(days=1)
                continue
            
            # Step 2: Process sentiment for each news article
            processed_tickers = set()
            for news in news_list:
                try:
                    news_id = news['id']
                    ticker = news['ticker']
                    processed_tickers.add(ticker)
                    
                    # Check if already processed
                    if is_sentiment_processed(ore_conn, news_id):
                        logger.debug(f"Sentiment already processed for news_id {news_id}, skipping")
                        continue
                    
                    # Combine title and summary for sentiment analysis
                    text = f"{news['title']} {news['summary']}".strip()
                    
                    # Calculate sentiment
                    logger.info(f"Calculating sentiment for {ticker} news_id {news_id}")
                    sentiment_data = analyzer.analyze_sentiment(text)
                    
                    # Insert sentiment
                    insert_sentiment(ore_conn, news_id, ticker, sentiment_data)
                    total_sentiments_processed += 1
                    
                    logger.info(f"✓ Processed sentiment for {ticker} news_id {news_id}: {sentiment_data['label']} ({sentiment_data['score']:.3f})")
                    
                except Exception as e:
                    total_errors += 1
                    logger.error(f"ERROR: Failed to process sentiment for news_id {news.get('id', 'unknown')}: {e}", exc_info=True)
                    continue
            
            # Step 3: Calculate aggregated sentiment signals for each ticker
            for ticker in processed_tickers:
                try:
                    # Check if signal already processed
                    if is_signal_processed(main_conn, ticker, current_date, 'SENTIMENT_YFINANCE_NEWS'):
                        logger.info(f"Signal already processed for {ticker} on {current_date}, skipping")
                        continue
                    
                    # Get sentiment for last 7 days
                    sentiment_list = get_sentiment_for_date_range(ore_conn, ticker, current_date, days=7)
                    
                    if not sentiment_list:
                        # No sentiment data available - insert NULL signal
                        logger.info(f"No sentiment data available for {ticker} on {current_date}, inserting NULL signal")
                        insert_signal(
                            main_conn,
                            ticker,
                            current_date,
                            'SENTIMENT_YFINANCE_NEWS',
                            None,  # NULL value
                            {'reason': 'no_sentiment_data_available', 'date': str(current_date)}
                        )
                        total_signals_processed += 1
                        logger.info(f"✓ Inserted NULL signal for {ticker} on {current_date} (no sentiment data)")
                    else:
                        # Calculate aggregated sentiment
                        aggregated = calculate_aggregated_sentiment(sentiment_list, current_date)
                        
                        # Insert signal
                        insert_signal(
                            main_conn,
                            ticker,
                            current_date,
                            'SENTIMENT_YFINANCE_NEWS',
                            aggregated['aggregated_score'],
                            aggregated['metadata']
                        )
                        
                        total_signals_processed += 1
                        logger.info(f"✓ Processed aggregated sentiment signal for {ticker} on {current_date}: {aggregated['aggregated_score']:.3f} (from {aggregated['count']} news articles)")
                    
                except Exception as e:
                    total_errors += 1
                    logger.error(f"ERROR: Failed to process signal for {ticker} on {current_date}: {e}", exc_info=True)
                    continue
        
        except Exception as e:
            total_errors += 1
            logger.error(f"ERROR: Failed to process date {current_date}: {e}", exc_info=True)
        
        # Move to next date
        current_date += timedelta(days=1)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Total sentiments processed: {total_sentiments_processed}")
    logger.info(f"Total signals processed: {total_signals_processed}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"{'='*60}\n")


def get_date_range_from_env() -> Tuple[date, date]:
    """Get date range from environment variables."""
    start_date_str = os.getenv('START_DATE')
    end_date_str = os.getenv('END_DATE')
    
    if start_date_str and end_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            logger.info(f"Using date range from environment: {start_date} to {end_date}")
            return start_date, end_date
        except ValueError as e:
            logger.error(f"Invalid date format in environment variables: {e}")
            logger.error("Expected format: YYYY-MM-DD")
            raise
    else:
        # Default to today if not specified
        today = date.today()
        logger.info(f"No date range specified, using today: {today}")
        return today, today


def main():
    """Main execution function."""
    ore_conn = None
    main_conn = None
    
    try:
        logger.info("="*60)
        logger.info("Starting news sentiment signal calculation...")
        logger.info("="*60)
        
        # Get date range
        start_date, end_date = get_date_range_from_env()
        
        # Connect to databases
        logger.info("Connecting to databases...")
        ore_conn = get_ore_db_connection()
        main_conn = get_main_db_connection()
        
        # Create sentiment table if needed
        logger.info("Creating/verifying sentiment table structure...")
        create_sentiment_table(ore_conn)
        
        # Ensure signal_raw.value column allows NULL
        logger.info("Ensuring signal_raw.value column allows NULL values...")
        ensure_value_column_nullable(main_conn)
        
        # Initialize FinBERT analyzer
        logger.info("Initializing FinBERT sentiment analyzer...")
        analyzer = FinBERTSentimentAnalyzer()
        
        # Process date range
        process_date_range(start_date, end_date, ore_conn, main_conn, analyzer)
        
        logger.info("Sentiment signal calculation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"FATAL ERROR during execution: {e}", exc_info=True)
        return 1
        
    finally:
        # Close connections
        if ore_conn:
            ore_conn.close()
            logger.info("Closed ORE database connection")
        if main_conn:
            main_conn.close()
            logger.info("Closed main database connection")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
