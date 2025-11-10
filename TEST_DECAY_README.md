# Decay Calculation Tests

This directory contains comprehensive tests for the decay calculation in news signal score computation.

## Test Files

1. **test_decay.py** - Unit tests for the decay calculation function
2. **test_decay_real_world.py** - Real-world scenario tests simulating actual usage

## Running the Tests

```bash
# Run basic decay tests
python test_decay.py

# Run real-world scenario tests
python test_decay_real_world.py
```

## Test Results

All tests pass, confirming that:
- ✓ Decay weights are correctly calculated (0.5^days_ago)
- ✓ Day 0 (most recent) has weight 1.0
- ✓ Day 1 has weight 0.5
- ✓ Day 2 has weight 0.25
- ✓ Day 6 has weight 0.015625
- ✓ News older than 7 days is excluded
- ✓ Future news is excluded
- ✓ Datetime objects are correctly converted to dates
- ✓ Multiple news items on the same day are correctly grouped

## Decay Formula

The decay calculation uses:
- **Weight = 0.5^days_ago**
- **Aggregated Score = Σ(score × weight) / Σ(weight)**

Where:
- `days_ago = (target_date - published_date).days`
- Only news within 7 days (0-6 days ago) is included

## Example

For target date 2024-01-10:
- News from 2024-01-10 (Day 0): weight = 1.0
- News from 2024-01-09 (Day 1): weight = 0.5
- News from 2024-01-08 (Day 2): weight = 0.25
- News from 2024-01-07 (Day 3): weight = 0.125
- News from 2024-01-06 (Day 4): weight = 0.0625
- News from 2024-01-05 (Day 5): weight = 0.03125
- News from 2024-01-04 (Day 6): weight = 0.015625
- News from 2024-01-03 (Day 7): **excluded** (>= 7 days)

## Findings

Based on comprehensive testing, the decay calculation logic is **working correctly**. All test cases pass, including:
- Basic decay calculation
- Same-day news handling
- Old news exclusion
- Future news exclusion
- Decay weight progression
- Datetime object handling
- Real-world multi-day scenarios

If decay appears not to be working in production, the issue may be:
1. Signals being cached/reused instead of recalculated
2. Incorrect date being passed to the function
3. Data retrieval issues (wrong date range)
4. Signal storage/retrieval not using the decay-calculated values

## Debugging

To debug decay issues:
1. Check the metadata in the signal_raw table - it should contain `decay_factor: 0.5` and `sentiment_by_day` breakdown
2. Verify that `target_date` passed to `calculate_aggregated_sentiment` is correct
3. Check that `get_sentiment_for_date_range` is retrieving the correct date range
4. Verify that signals are being recalculated for each date, not reused


