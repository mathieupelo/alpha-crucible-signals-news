# Alpha Crucible Signals - News

Fetches data from ORE database, calculates signal scores, and stores them in the main database.

## Overview

This repository is responsible for:
1. Fetching news data from ORE database
2. Calculating signal scores from the data (sentiment, relevance, etc.)
3. Storing signal scores in the main database

## Setup

1. Clone this repository
2. Copy `.env.example` to `.env` and fill in your credentials
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python main.py`

## Docker

Build and run with Docker:

```bash
docker build -t alpha-crucible-signals-news .
docker run --env-file .env alpha-crucible-signals-news
```

## Environment Variables

See `.env.example` for required environment variables.

## Exit Codes

- `0`: Success
- `1`: Failure

## Testing

Run all tests:
```bash
python run_tests.py
```

Run individual test files:
```bash
python test_decay.py          # Tests decay calculation logic (no database)
python test_integration.py    # Tests database integration and cleanup
```

### Test Cleanup

All tests that insert data into the database automatically clean up after themselves using the `TestDataCleanup` class in `test_cleanup.py`. This ensures:

- No test data remains in the database after tests complete
- Tests can be run multiple times without conflicts
- Database state is preserved

The cleanup system tracks:
- Signals inserted into `signal_raw`
- Sentiments inserted into `copper.yfinance_news_sentiment`
- Signal records created in `signal` table (if applicable)

Cleanup happens automatically in `finally` blocks, even if tests fail.

## Development

This is a skeleton repository. Implement your signal calculation logic in `main.py`.

