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
from psycopg2 import errors as psycopg2_errors
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


def get_unprocessed_news_in_range(ore_conn, end_date: date, days: int = 28) -> List[Dict[str, Any]]:
    """Get all news articles from the last N days that haven't had sentiment calculated yet."""
    try:
        start_date = end_date - timedelta(days=days-1)
        with ore_conn.cursor() as cursor:
            cursor.execute("""
                SELECT n.id, n.ticker, n.title, n.summary, n.publisher, n.link, n.published_date, n.image_url
                FROM copper.yfinance_news n
                LEFT JOIN copper.yfinance_news_sentiment s ON n.id = s.yfinance_news_id
                WHERE CAST(n.published_date AS DATE) >= %s
                AND CAST(n.published_date AS DATE) <= %s
                AND s.yfinance_news_id IS NULL
                ORDER BY n.published_date DESC, n.ticker;
            """, (start_date, end_date))
            
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
            
            logger.info(f"Found {len(news_list)} unprocessed news articles from {start_date} to {end_date}")
            return news_list
            
    except Exception as e:
        logger.error(f"Error fetching unprocessed news from {start_date} to {end_date}: {e}")
        raise


def get_news_for_ticker_and_date(ore_conn, ticker: str, target_date: date) -> List[Dict[str, Any]]:
    """Get all news articles for a specific ticker and date from ORE database."""
    try:
        with ore_conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, ticker, title, summary, publisher, link, published_date, image_url
                FROM copper.yfinance_news
                WHERE ticker = %s
                AND CAST(published_date AS DATE) = %s
                ORDER BY published_date;
            """, (ticker, target_date))
            
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
            
            return news_list
            
    except Exception as e:
        logger.error(f"Error fetching news for {ticker} on {target_date}: {e}")
        raise


def get_news_for_date(ore_conn, target_date: date) -> List[Dict[str, Any]]:
    """Get all news articles for a specific date from ORE database."""
    try:
        with ore_conn.cursor() as cursor:
            # Use CAST to date for more reliable date comparison
            # This handles timezone issues better than DATE() function
            cursor.execute("""
                SELECT id, ticker, title, summary, publisher, link, published_date, image_url
                FROM copper.yfinance_news
                WHERE CAST(published_date AS DATE) = %s
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
            
            # If no news found, check if there's any news in the database and suggest recent dates
            if len(news_list) == 0:
                cursor.execute("""
                    SELECT COUNT(*) as total, 
                           MIN(CAST(published_date AS DATE)) as min_date,
                           MAX(CAST(published_date AS DATE)) as max_date
                    FROM copper.yfinance_news;
                """)
                stats = cursor.fetchone()
                if stats and stats[0] > 0:
                    logger.warning(
                        f"No news found for {target_date}, but database contains {stats[0]} total articles "
                        f"with date range {stats[1]} to {stats[2]}"
                    )
                    # Suggest recent dates with news
                    cursor.execute("""
                        SELECT CAST(published_date AS DATE) as pub_date, COUNT(*) as count
                        FROM copper.yfinance_news
                        WHERE CAST(published_date AS DATE) >= %s
                        GROUP BY CAST(published_date AS DATE)
                        ORDER BY pub_date DESC
                        LIMIT 5
                    """, (target_date - timedelta(days=7),))
                    recent_dates = cursor.fetchall()
                    if recent_dates:
                        logger.info("Recent dates with news (consider processing these instead):")
                        for pub_date, count in recent_dates:
                            logger.info(f"  {pub_date}: {count} articles")
                else:
                    logger.warning(f"No news found for {target_date} and database appears empty")
            
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


def is_signal_processed(main_conn, ticker: str, target_date: date, signal_name: str, company_uid: Optional[str] = None) -> bool:
    """
    Check if signal has already been calculated for a company_uid/date or ticker/date.
    Prioritizes company_uid check if available to avoid duplicate processing.
    """
    try:
        with main_conn.cursor() as cursor:
            # Check if company_uid column exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'signal_raw' AND column_name = 'company_uid'
                )
            """)
            has_company_uid = cursor.fetchone()[0]
            
            # Check if signal_id column exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'signal_raw' AND column_name = 'signal_id'
                )
            """)
            has_signal_id = cursor.fetchone()[0]
            
            # Resolve company_uid if not provided but column exists
            if has_company_uid and not company_uid:
                company_uid = resolve_ticker_to_company_uid(main_conn, ticker)
            
            # Use company_uid for checking if available (more efficient, avoids duplicates)
            if has_company_uid and company_uid:
                if has_signal_id:
                    signal_id = get_or_create_signal_id(main_conn, signal_name)
                    if signal_id:
                        cursor.execute("""
                            SELECT COUNT(*) 
                            FROM signal_raw 
                            WHERE company_uid = %s 
                            AND asof_date = %s 
                            AND signal_id = %s;
                        """, (company_uid, target_date, signal_id))
                    else:
                        cursor.execute("""
                            SELECT COUNT(*) 
                            FROM signal_raw 
                            WHERE company_uid = %s 
                            AND asof_date = %s 
                            AND signal_name = %s;
                        """, (company_uid, target_date, signal_name))
                else:
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM signal_raw 
                        WHERE company_uid = %s 
                        AND asof_date = %s 
                        AND signal_name = %s;
                    """, (company_uid, target_date, signal_name))
            else:
                # Fallback to ticker-based check
                if has_signal_id:
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
                        cursor.execute("""
                            SELECT COUNT(*) 
                            FROM signal_raw 
                            WHERE ticker = %s 
                            AND asof_date = %s 
                            AND signal_name = %s;
                        """, (ticker, target_date, signal_name))
                else:
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


def get_all_tickers_with_company_uid(main_conn) -> Dict[str, str]:
    """
    Get all tickers with their corresponding company_uid from varrock.tickers table.
    
    Returns:
        Dictionary mapping ticker -> company_uid
    """
    try:
        with main_conn.cursor() as cursor:
            # Query all tickers with company_uid from varrock.tickers table
            # Filter by is_active = TRUE to only get active tickers
            # Use yfinance_symbol if available, otherwise use ticker
            cursor.execute("""
                SELECT DISTINCT 
                    COALESCE(t.yfinance_symbol, t.ticker) as ticker,
                    t.company_uid
                FROM varrock.tickers t
                WHERE t.is_active = TRUE
                ORDER BY ticker;
            """)
            ticker_map = {}
            for row in cursor.fetchall():
                ticker = row[0]
                company_uid = row[1]
                # If multiple tickers map to same company, keep the first one
                if ticker not in ticker_map:
                    ticker_map[ticker] = company_uid
            
            logger.info(f"Found {len(ticker_map)} distinct tickers with company_uid (from varrock.tickers)")
            return ticker_map
    except Exception as e:
        logger.error(f"Error fetching tickers with company_uid: {e}")
        raise


def get_all_tickers(main_conn) -> Set[str]:
    """Get all distinct tickers from varrock.tickers table (all tickers, even if not in universes)."""
    ticker_map = get_all_tickers_with_company_uid(main_conn)
    return set(ticker_map.keys())


def resolve_ticker_to_company_uid(main_conn, ticker: str) -> Optional[str]:
    """
    Resolve a ticker symbol to company_uid via varrock.tickers.
    
    Args:
        main_conn: Database connection
        ticker: Ticker symbol to resolve (can be either ticker or yfinance_symbol)
        
    Returns:
        company_uid if found, None if not found
    """
    try:
        with main_conn.cursor() as cursor:
            # Try to find company_uid by matching either ticker or yfinance_symbol
            cursor.execute("""
                SELECT company_uid 
                FROM varrock.tickers 
                WHERE ticker = %s OR yfinance_symbol = %s
                LIMIT 1
            """, (ticker.strip().upper(), ticker.strip().upper()))
            
            result = cursor.fetchone()
            if result:
                return str(result[0])
            return None
    except Exception as e:
        logger.warning(f"Error resolving ticker {ticker} to company_uid: {e}")
        return None


def get_tickers_with_sentiment_in_range(ore_conn, end_date: date, days: int = 28) -> Set[str]:
    """Get all distinct tickers that have sentiment data in the last N days."""
    try:
        start_date = end_date - timedelta(days=days-1)
        
        with ore_conn.cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT s.ticker
                FROM copper.yfinance_news_sentiment s
                JOIN copper.yfinance_news n ON s.yfinance_news_id = n.id
                WHERE DATE(n.published_date) >= %s
                AND DATE(n.published_date) <= %s
                ORDER BY s.ticker;
            """, (start_date, end_date))
            
            tickers = {row[0] for row in cursor.fetchall()}
            logger.debug(f"Found {len(tickers)} tickers with sentiment data from {start_date} to {end_date}")
            return tickers
    except Exception as e:
        logger.error(f"Error fetching tickers with sentiment in range: {e}")
        return set()


def get_sentiment_for_date_range(ore_conn, ticker: str, end_date: date, days: int = 28) -> List[Dict[str, Any]]:
    """
    Get sentiment scores for a ticker over the last N days up to and including the end_date.
    So if end_date is 2025-11-09 and days=28, it gets sentiment from 2025-10-13 to 2025-11-09 (inclusive).
    """
    try:
        # Calculate start_date: end_date - (days-1) to get exactly 'days' days including end_date
        start_date = end_date - timedelta(days=days-1)
        # Include end_date (current date)
        
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
                AND CAST(n.published_date AS DATE) >= %s
                AND CAST(n.published_date AS DATE) <= %s
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


def calculate_aggregated_sentiment(sentiment_list: List[Dict[str, Any]], target_date: date, window_days: int = 28) -> Dict[str, Any]:
    """
    Calculate aggregated sentiment with decay weighting over a specified window.
    
    Decay: Day 0 (most recent) = 1.0, Day 1 = 0.5, Day 2 = 0.25, etc.
    Window: Default 28 days (news older than window_days is excluded).
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
        
        if days_ago < 0 or days_ago >= window_days:
            continue  # Skip if outside window
        
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
        'calculation_method': f'{window_days}_day_decay_weighted_average',
        'decay_factor': 0.5,
        'window_days': window_days
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
            
            # Check if company_uid column exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'signal_raw' AND column_name = 'company_uid'
                )
            """)
            has_company_uid = cursor.fetchone()[0]
            
            # Resolve ticker to company_uid if column exists
            company_uid = None
            if has_company_uid:
                company_uid = resolve_ticker_to_company_uid(main_conn, ticker)
                if not company_uid:
                    logger.warning(f"Could not resolve ticker {ticker} to company_uid, inserting with NULL company_uid")
            
            # Prepare metadata (can be None)
            metadata_json = Json(metadata) if metadata else None
            
            if has_signal_id:
                # Get or create signal_id
                signal_id = get_or_create_signal_id(main_conn, signal_name)
                if signal_id:
                    if has_company_uid and company_uid:
                        # Use signal_id and company_uid in insert - prioritize company_uid for conflict resolution
                        # Try to use company_uid-based conflict first if unique constraint exists
                        try:
                            cursor.execute("""
                                INSERT INTO signal_raw 
                                (asof_date, ticker, signal_name, signal_id, company_uid, value, metadata)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (asof_date, ticker, signal_id) 
                                DO UPDATE SET
                                    company_uid = EXCLUDED.company_uid,
                                    value = EXCLUDED.value,
                                    metadata = EXCLUDED.metadata,
                                    created_at = CURRENT_TIMESTAMP;
                            """, (
                                target_date,
                                ticker,
                                signal_name,
                                signal_id,
                                company_uid,
                                value,  # Can be None
                                metadata_json
                            ))
                        except psycopg2_errors.UniqueViolation:
                            # If company_uid-based conflict, try alternative conflict resolution
                            cursor.execute("""
                                INSERT INTO signal_raw 
                                (asof_date, ticker, signal_name, signal_id, company_uid, value, metadata)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (asof_date, ticker, signal_name) 
                                DO UPDATE SET
                                    signal_id = EXCLUDED.signal_id,
                                    company_uid = EXCLUDED.company_uid,
                                    value = EXCLUDED.value,
                                    metadata = EXCLUDED.metadata,
                                    created_at = CURRENT_TIMESTAMP;
                            """, (
                                target_date,
                                ticker,
                                signal_name,
                                signal_id,
                                company_uid,
                                value,  # Can be None
                                metadata_json
                            ))
                    elif has_company_uid:
                        # company_uid column exists but couldn't resolve - use ticker-based conflict
                        cursor.execute("""
                            INSERT INTO signal_raw 
                            (asof_date, ticker, signal_name, signal_id, company_uid, value, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (asof_date, ticker, signal_id) 
                            DO UPDATE SET
                                company_uid = EXCLUDED.company_uid,
                                value = EXCLUDED.value,
                                metadata = EXCLUDED.metadata,
                                created_at = CURRENT_TIMESTAMP;
                        """, (
                            target_date,
                            ticker,
                            signal_name,
                            signal_id,
                            company_uid,  # NULL
                            value,  # Can be None
                            metadata_json
                        ))
                    else:
                        # Use signal_id without company_uid
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
                    # Fallback: try without signal_id
                    if has_company_uid:
                        cursor.execute("""
                            INSERT INTO signal_raw 
                            (asof_date, ticker, signal_name, company_uid, value, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (asof_date, ticker, signal_name) 
                            DO UPDATE SET
                                company_uid = EXCLUDED.company_uid,
                                value = EXCLUDED.value,
                                metadata = EXCLUDED.metadata,
                                created_at = CURRENT_TIMESTAMP;
                        """, (
                            target_date,
                            ticker,
                            signal_name,
                            company_uid,  # Can be None
                            value,  # Can be None
                            metadata_json
                        ))
                    else:
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
                if has_company_uid:
                    cursor.execute("""
                        INSERT INTO signal_raw 
                        (asof_date, ticker, signal_name, company_uid, value, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (asof_date, ticker, signal_name) 
                        DO UPDATE SET
                            company_uid = EXCLUDED.company_uid,
                            value = EXCLUDED.value,
                            metadata = EXCLUDED.metadata,
                            created_at = CURRENT_TIMESTAMP;
                    """, (
                        target_date,
                        ticker,
                        signal_name,
                        company_uid,  # Can be None
                        value,  # Can be None
                        metadata_json
                    ))
                else:
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
    """
    Process sentiment calculation for all dates in the range.
    
    Expected behavior:
    1. Get all tickers with company_uid from varrock.tickers
    2. For each date in range:
       2.1 For each ticker:
           2.1.1 Get news for that ticker for that day
           2.1.2 Calculate sentiment using FinBERT
           2.1.3 Insert/replace sentiment in copper.yfinance_news_sentiment
    3. For each date in range:
       3.1 For each ticker:
           3.1.1 Get sentiment scores for last 28 days before current date
           3.1.2 Aggregate with decay of 0.5 per day
       3.2 Insert aggregated values in signal_raw with company_uid
    """
    # Step 1: Get all tickers with company_uid
    logger.info("Step 1: Getting all tickers with company_uid from varrock.tickers...")
    ticker_to_company_uid = get_all_tickers_with_company_uid(main_conn)
    all_tickers = list(ticker_to_company_uid.keys())
    logger.info(f"Found {len(all_tickers)} tickers to process")
    
    total_sentiments_processed = 0
    total_signals_processed = 0
    total_errors = 0
    
    # Step 2: Process sentiment for each date and ticker
    logger.info(f"\n{'='*60}")
    logger.info("Step 2: Processing sentiment for news articles")
    logger.info(f"{'='*60}")
    
    current_date = start_date
    while current_date <= end_date:
        logger.info(f"\nProcessing sentiment for date: {current_date}")
        
        date_sentiments = 0
        date_news_count = 0
        
        for ticker in all_tickers:
            try:
                # Step 2.1.1: Get all news for this ticker for this day
                news_list = get_news_for_ticker_and_date(ore_conn, ticker, current_date)
                
                if not news_list:
                    logger.debug(f"  {ticker}: No news for this date")
                    continue  # No news for this ticker on this date
                
                logger.info(f"  {ticker}: Found {len(news_list)} news articles")
                date_news_count += len(news_list)
                
                # Step 2.1.2 & 2.1.3: Calculate sentiment and insert/replace
                for news in news_list:
                    try:
                        news_id = news['id']
                        
                        # Combine title and summary for sentiment analysis
                        text = f"{news['title']} {news['summary']}".strip()
                        
                        if not text:
                            logger.warning(f"    Skipping news_id {news_id} - no text content")
                            continue
                        
                        # Calculate sentiment using FinBERT
                        logger.debug(f"    Calculating sentiment for {ticker} news_id {news_id}")
                        sentiment_data = analyzer.analyze_sentiment(text)
                        
                        # Insert or replace sentiment (insert_sentiment handles ON CONFLICT DO UPDATE)
                        insert_sentiment(ore_conn, news_id, ticker, sentiment_data)
                        total_sentiments_processed += 1
                        date_sentiments += 1
                        
                        logger.debug(f"    ✓ Processed sentiment for {ticker} news_id {news_id}: {sentiment_data['label']} ({sentiment_data['score']:.3f})")
                        
                    except Exception as e:
                        total_errors += 1
                        logger.error(f"    ERROR: Failed to process sentiment for {ticker} news_id {news.get('id', 'unknown')}: {e}", exc_info=True)
                        continue
                        
            except Exception as e:
                total_errors += 1
                logger.error(f"  ERROR: Failed to process news for {ticker} on {current_date}: {e}", exc_info=True)
                continue
        
        logger.info(f"  Date {current_date} summary: {date_news_count} news articles, {date_sentiments} sentiments processed")
        current_date += timedelta(days=1)
    
    # Step 3: Calculate aggregated signals for each date and ticker
    logger.info(f"\n{'='*60}")
    logger.info("Step 3: Calculating aggregated sentiment signals")
    logger.info(f"{'='*60}")
    
    current_date = start_date
    while current_date <= end_date:
        logger.info(f"\nProcessing aggregated signals for date: {current_date}")
        
        # Track processed company_uid pairs to avoid duplicates
        processed_company_uids = set()
        
        tickers_processed_count = 0
        tickers_skipped_count = 0
        tickers_null_count = 0
        tickers_with_sentiment_count = 0
        
        for ticker in all_tickers:
            try:
                company_uid = ticker_to_company_uid.get(ticker)
                
                # Check if this company_uid/signal_name pair has already been processed
                if company_uid and (company_uid, 'SENTIMENT_YFINANCE_NEWS') in processed_company_uids:
                    logger.debug(f"  {ticker}: Signal already processed for company_uid {company_uid}, skipping")
                    tickers_skipped_count += 1
                    continue
                
                # Check if signal already processed in database
                if is_signal_processed(main_conn, ticker, current_date, 'SENTIMENT_YFINANCE_NEWS', company_uid):
                    logger.info(f"  {ticker}: Signal already processed in database, skipping (use --force to reprocess)")
                    tickers_skipped_count += 1
                    continue
                
                # Step 3.1.1: Get sentiment scores for last 28 days up to and including current date
                sentiment_list = get_sentiment_for_date_range(ore_conn, ticker, current_date, days=28)
                
                if not sentiment_list:
                    # No sentiment data available - insert NULL signal
                    logger.info(f"  {ticker}: No sentiment data available, inserting NULL signal")
                    tickers_null_count += 1
                    insert_signal(
                        main_conn,
                        ticker,
                        current_date,
                        'SENTIMENT_YFINANCE_NEWS',
                        None,  # NULL value
                        {'reason': 'no_sentiment_data_available', 'date': str(current_date)}
                    )
                    
                    # Mark this company_uid/signal_name pair as processed
                    if company_uid:
                        processed_company_uids.add((company_uid, 'SENTIMENT_YFINANCE_NEWS'))
                    
                    total_signals_processed += 1
                else:
                    # Step 3.1.2: Aggregate sentiment with decay of 0.5 per day
                    aggregated = calculate_aggregated_sentiment(sentiment_list, current_date, window_days=28)
                    
                    # Step 3.2: Insert aggregated value in signal_raw with company_uid
                    insert_signal(
                        main_conn,
                        ticker,
                        current_date,
                        'SENTIMENT_YFINANCE_NEWS',
                        aggregated['aggregated_score'],
                        aggregated['metadata']
                    )
                    
                    # Mark this company_uid/signal_name pair as processed
                    if company_uid:
                        processed_company_uids.add((company_uid, 'SENTIMENT_YFINANCE_NEWS'))
                    
                    total_signals_processed += 1
                    tickers_with_sentiment_count += 1
                    logger.info(f"  {ticker}: Aggregated score = {aggregated['aggregated_score']:.3f} (from {aggregated['count']} news articles)")
                    
            except Exception as e:
                total_errors += 1
                logger.error(f"  ERROR: Failed to process signal for {ticker} on {current_date}: {e}", exc_info=True)
                continue
        
        logger.info(f"  Processed {tickers_with_sentiment_count} tickers with sentiment, {tickers_null_count} with NULL signals, {tickers_skipped_count} skipped")
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
        # Default to yesterday if not specified (more likely to have news)
        yesterday = date.today() - timedelta(days=1)
        logger.info(f"No date range specified, using yesterday: {yesterday} (today may not have news yet)")
        return yesterday, yesterday


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
