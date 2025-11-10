#!/usr/bin/env python3
"""
Test for decay calculation in news signal score calculation.
"""

import sys
from datetime import date, datetime, timedelta
from typing import List, Dict, Any

# Import the function we want to test
sys.path.insert(0, '.')
from main import calculate_aggregated_sentiment


def test_decay_calculation():
    """Test that decay is correctly applied to sentiment scores."""
    
    # Test case 1: News from different days with known scores
    # Target date: 2024-01-10
    target_date = date(2024, 1, 10)
    
    # Create sentiment data:
    # - Day 0 (2024-01-10): score = 1.0 (should have weight 1.0)
    # - Day 1 (2024-01-09): score = 0.5 (should have weight 0.5)
    # - Day 2 (2024-01-08): score = 0.25 (should have weight 0.25)
    # - Day 3 (2024-01-07): score = 0.0 (should have weight 0.125)
    
    sentiment_list = [
        {
            'published_date': date(2024, 1, 10),  # Day 0
            'sentiment_score': 1.0
        },
        {
            'published_date': date(2024, 1, 9),   # Day 1
            'sentiment_score': 0.5
        },
        {
            'published_date': date(2024, 1, 8),   # Day 2
            'sentiment_score': 0.25
        },
        {
            'published_date': date(2024, 1, 7),   # Day 3
            'sentiment_score': 0.0
        }
    ]
    
    result = calculate_aggregated_sentiment(sentiment_list, target_date)
    
    print("=" * 60)
    print("Test 1: Basic decay calculation")
    print("=" * 60)
    print(f"Target date: {target_date}")
    print(f"Number of news items: {len(sentiment_list)}")
    print(f"Aggregated score: {result['aggregated_score']:.6f}")
    print(f"Total weight: {result['metadata']['total_weight']:.6f}")
    print(f"\nExpected calculation:")
    print(f"  Day 0 (2024-01-10): score=1.0, weight=1.0, weighted=1.0")
    print(f"  Day 1 (2024-01-09): score=0.5, weight=0.5, weighted=0.25")
    print(f"  Day 2 (2024-01-08): score=0.25, weight=0.25, weighted=0.0625")
    print(f"  Day 3 (2024-01-07): score=0.0, weight=0.125, weighted=0.0")
    print(f"  Total weighted: 1.3125")
    print(f"  Total weight: 1.875")
    print(f"  Expected aggregated: 1.3125 / 1.875 = 0.7")
    print(f"\nActual aggregated: {result['aggregated_score']:.6f}")
    
    # Manual calculation
    expected_weighted = (1.0 * 1.0) + (0.5 * 0.5) + (0.25 * 0.25) + (0.0 * 0.125)
    expected_total_weight = 1.0 + 0.5 + 0.25 + 0.125
    expected_aggregated = expected_weighted / expected_total_weight
    
    print(f"\nExpected weighted sum: {expected_weighted:.6f}")
    print(f"Expected total weight: {expected_total_weight:.6f}")
    print(f"Expected aggregated: {expected_aggregated:.6f}")
    
    # Check if decay is working
    if abs(result['aggregated_score'] - expected_aggregated) < 0.0001:
        print("\n✓ PASS: Decay calculation is correct")
    else:
        print(f"\n✗ FAIL: Decay calculation is incorrect!")
        print(f"  Difference: {abs(result['aggregated_score'] - expected_aggregated):.6f}")
    
    # Print detailed breakdown
    print(f"\nDetailed breakdown by day:")
    for days_ago, day_data in sorted(result['metadata']['sentiment_by_day'].items()):
        pub_date = target_date - timedelta(days=days_ago)
        expected_weight = 0.5 ** days_ago
        print(f"  Day {days_ago} ({pub_date}):")
        print(f"    Count: {day_data['count']}")
        print(f"    Total weight: {day_data['total_weight']:.6f} (expected: {expected_weight:.6f})")
        if abs(day_data['total_weight'] - expected_weight) > 0.0001:
            print(f"    ✗ Weight mismatch!")
        else:
            print(f"    ✓ Weight correct")
    
    return result


def test_decay_with_same_day_news():
    """Test that same-day news gets full weight."""
    
    target_date = date(2024, 1, 10)
    
    # All news from the same day (target_date)
    sentiment_list = [
        {
            'published_date': date(2024, 1, 10),  # Day 0
            'sentiment_score': 0.8
        },
        {
            'published_date': date(2024, 1, 10),  # Day 0
            'sentiment_score': 0.6
        },
        {
            'published_date': date(2024, 1, 10),  # Day 0
            'sentiment_score': 0.4
        }
    ]
    
    result = calculate_aggregated_sentiment(sentiment_list, target_date)
    
    print("\n" + "=" * 60)
    print("Test 2: Same-day news (should all have weight 1.0)")
    print("=" * 60)
    print(f"Target date: {target_date}")
    print(f"Number of news items: {len(sentiment_list)}")
    print(f"Aggregated score: {result['aggregated_score']:.6f}")
    print(f"Total weight: {result['metadata']['total_weight']:.6f}")
    
    # All should have weight 1.0, so aggregated should be simple average
    expected_aggregated = (0.8 + 0.6 + 0.4) / 3.0
    expected_total_weight = 3.0  # 3 items * 1.0 weight each
    
    print(f"\nExpected aggregated (simple average): {expected_aggregated:.6f}")
    print(f"Expected total weight: {expected_total_weight:.6f}")
    print(f"Actual aggregated: {result['aggregated_score']:.6f}")
    print(f"Actual total weight: {result['metadata']['total_weight']:.6f}")
    
    if abs(result['aggregated_score'] - expected_aggregated) < 0.0001:
        print("\n✓ PASS: Same-day news correctly weighted")
    else:
        print(f"\n✗ FAIL: Same-day news incorrectly weighted!")
        print(f"  Difference: {abs(result['aggregated_score'] - expected_aggregated):.6f}")
    
    return result


def test_decay_with_old_news_excluded():
    """Test that news older than 7 days is excluded."""
    
    target_date = date(2024, 1, 10)
    
    # Mix of recent and old news
    sentiment_list = [
        {
            'published_date': date(2024, 1, 10),  # Day 0 - should be included
            'sentiment_score': 1.0
        },
        {
            'published_date': date(2024, 1, 3),   # Day 7 - should be EXCLUDED (>= 7)
            'sentiment_score': 100.0  # Very high score to make it obvious if included
        },
        {
            'published_date': date(2024, 1, 2),   # Day 8 - should be EXCLUDED
            'sentiment_score': 100.0
        },
        {
            'published_date': date(2024, 1, 9),   # Day 1 - should be included
            'sentiment_score': 0.5
        }
    ]
    
    result = calculate_aggregated_sentiment(sentiment_list, target_date)
    
    print("\n" + "=" * 60)
    print("Test 3: Old news exclusion (>= 7 days should be excluded)")
    print("=" * 60)
    print(f"Target date: {target_date}")
    print(f"Number of news items: {len(sentiment_list)}")
    print(f"Aggregated score: {result['aggregated_score']:.6f}")
    print(f"Weighted news count: {result['metadata']['weighted_news_count']}")
    
    # Should only include Day 0 and Day 1 news
    expected_weighted_count = 2
    expected_aggregated = (1.0 * 1.0 + 0.5 * 0.5) / (1.0 + 0.5)
    
    print(f"\nExpected weighted news count: {expected_weighted_count}")
    print(f"Expected aggregated: {expected_aggregated:.6f}")
    print(f"Actual weighted news count: {result['metadata']['weighted_news_count']}")
    print(f"Actual aggregated: {result['aggregated_score']:.6f}")
    
    if result['metadata']['weighted_news_count'] == expected_weighted_count:
        print("\n✓ PASS: Old news correctly excluded")
    else:
        print(f"\n✗ FAIL: Old news not correctly excluded!")
        print(f"  Expected {expected_weighted_count}, got {result['metadata']['weighted_news_count']}")
    
    # Check that aggregated score is reasonable (should be around 0.833, not 100+)
    if result['aggregated_score'] < 10.0:
        print("✓ PASS: Aggregated score is reasonable (old news excluded)")
    else:
        print(f"✗ FAIL: Aggregated score is too high, suggesting old news was included!")
    
    return result


def test_decay_with_future_news_excluded():
    """Test that future news is excluded."""
    
    target_date = date(2024, 1, 10)
    
    # Mix of past and future news
    sentiment_list = [
        {
            'published_date': date(2024, 1, 10),  # Day 0 - should be included
            'sentiment_score': 1.0
        },
        {
            'published_date': date(2024, 1, 11),   # Future - should be EXCLUDED
            'sentiment_score': 100.0
        },
        {
            'published_date': date(2024, 1, 9),   # Day 1 - should be included
            'sentiment_score': 0.5
        }
    ]
    
    result = calculate_aggregated_sentiment(sentiment_list, target_date)
    
    print("\n" + "=" * 60)
    print("Test 4: Future news exclusion")
    print("=" * 60)
    print(f"Target date: {target_date}")
    print(f"Number of news items: {len(sentiment_list)}")
    print(f"Aggregated score: {result['aggregated_score']:.6f}")
    print(f"Weighted news count: {result['metadata']['weighted_news_count']}")
    
    # Should only include Day 0 and Day 1 news
    expected_weighted_count = 2
    expected_aggregated = (1.0 * 1.0 + 0.5 * 0.5) / (1.0 + 0.5)
    
    print(f"\nExpected weighted news count: {expected_weighted_count}")
    print(f"Expected aggregated: {expected_aggregated:.6f}")
    print(f"Actual weighted news count: {result['metadata']['weighted_news_count']}")
    print(f"Actual aggregated: {result['aggregated_score']:.6f}")
    
    if result['metadata']['weighted_news_count'] == expected_weighted_count:
        print("\n✓ PASS: Future news correctly excluded")
    else:
        print(f"\n✗ FAIL: Future news not correctly excluded!")
        print(f"  Expected {expected_weighted_count}, got {result['metadata']['weighted_news_count']}")
    
    return result


def test_decay_progression():
    """Test that decay weights decrease correctly as days_ago increases."""
    
    target_date = date(2024, 1, 10)
    
    # News from each day in the 7-day window, all with same score
    sentiment_list = []
    for i in range(7):
        sentiment_list.append({
            'published_date': target_date - timedelta(days=i),
            'sentiment_score': 1.0  # Same score for all
        })
    
    result = calculate_aggregated_sentiment(sentiment_list, target_date)
    
    print("\n" + "=" * 60)
    print("Test 5: Decay weight progression (all same score)")
    print("=" * 60)
    print(f"Target date: {target_date}")
    print(f"Number of news items: {len(sentiment_list)}")
    print(f"Aggregated score: {result['aggregated_score']:.6f}")
    
    print(f"\nWeight progression (should be 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):")
    for days_ago in sorted(result['metadata']['sentiment_by_day'].keys()):
        day_data = result['metadata']['sentiment_by_day'][days_ago]
        expected_weight = 0.5 ** days_ago
        actual_weight_per_item = day_data['total_weight'] / day_data['count']
        print(f"  Day {days_ago}: weight={actual_weight_per_item:.6f} (expected: {expected_weight:.6f})")
        if abs(actual_weight_per_item - expected_weight) < 0.0001:
            print(f"    ✓ Correct")
        else:
            print(f"    ✗ Incorrect!")
    
    # Since all scores are 1.0, aggregated should be weighted average of 1.0s = 1.0
    # But the weighted average should still be 1.0 regardless of weights
    if abs(result['aggregated_score'] - 1.0) < 0.0001:
        print("\n✓ PASS: Aggregated score is correct (1.0)")
    else:
        print(f"\n✗ FAIL: Aggregated score should be 1.0, got {result['aggregated_score']:.6f}")
    
    return result


def test_decay_with_datetime_objects():
    """Test that datetime objects are correctly handled (as they come from database)."""
    
    target_date = date(2024, 1, 10)
    
    # Simulate what comes from database - datetime objects with time components
    sentiment_list = [
        {
            'published_date': datetime(2024, 1, 10, 14, 30, 0),  # Day 0, with time
            'sentiment_score': 1.0
        },
        {
            'published_date': datetime(2024, 1, 9, 10, 15, 0),   # Day 1, with time
            'sentiment_score': 0.5
        },
        {
            'published_date': datetime(2024, 1, 8, 8, 0, 0),    # Day 2, with time
            'sentiment_score': 0.25
        }
    ]
    
    result = calculate_aggregated_sentiment(sentiment_list, target_date)
    
    print("\n" + "=" * 60)
    print("Test 6: Datetime objects with time components")
    print("=" * 60)
    print(f"Target date: {target_date}")
    print(f"Number of news items: {len(sentiment_list)}")
    print(f"Aggregated score: {result['aggregated_score']:.6f}")
    print(f"Total weight: {result['metadata']['total_weight']:.6f}")
    
    # Should be same as if we used date objects
    expected_aggregated = (1.0 * 1.0 + 0.5 * 0.5 + 0.25 * 0.25) / (1.0 + 0.5 + 0.25)
    
    print(f"\nExpected aggregated: {expected_aggregated:.6f}")
    print(f"Actual aggregated: {result['aggregated_score']:.6f}")
    
    if abs(result['aggregated_score'] - expected_aggregated) < 0.0001:
        print("\n✓ PASS: Datetime objects correctly handled")
    else:
        print(f"\n✗ FAIL: Datetime objects not correctly handled!")
        print(f"  Difference: {abs(result['aggregated_score'] - expected_aggregated):.6f}")
    
    # Check that days_ago calculation is correct
    print(f"\nDays ago calculation:")
    for days_ago, day_data in sorted(result['metadata']['sentiment_by_day'].items()):
        expected_weight = 0.5 ** days_ago
        actual_weight_per_item = day_data['total_weight'] / day_data['count']
        print(f"  Day {days_ago}: weight={actual_weight_per_item:.6f} (expected: {expected_weight:.6f})")
        if abs(actual_weight_per_item - expected_weight) < 0.0001:
            print(f"    ✓ Correct")
        else:
            print(f"    ✗ Incorrect!")
    
    return result


def test_decay_edge_case_same_date_different_times():
    """Test edge case: multiple news items on same date but different times."""
    
    target_date = date(2024, 1, 10)
    
    # Multiple news items on the same date but different times
    # All should have days_ago = 0 and weight = 1.0
    sentiment_list = [
        {
            'published_date': datetime(2024, 1, 10, 0, 0, 0),   # Midnight
            'sentiment_score': 0.8
        },
        {
            'published_date': datetime(2024, 1, 10, 12, 0, 0),  # Noon
            'sentiment_score': 0.6
        },
        {
            'published_date': datetime(2024, 1, 10, 23, 59, 59),  # End of day
            'sentiment_score': 0.4
        }
    ]
    
    result = calculate_aggregated_sentiment(sentiment_list, target_date)
    
    print("\n" + "=" * 60)
    print("Test 7: Same date, different times (all should be Day 0)")
    print("=" * 60)
    print(f"Target date: {target_date}")
    print(f"Number of news items: {len(sentiment_list)}")
    print(f"Aggregated score: {result['aggregated_score']:.6f}")
    print(f"Total weight: {result['metadata']['total_weight']:.6f}")
    
    # All should be Day 0 with weight 1.0
    expected_aggregated = (0.8 + 0.6 + 0.4) / 3.0
    expected_total_weight = 3.0
    
    print(f"\nExpected aggregated (simple average): {expected_aggregated:.6f}")
    print(f"Expected total weight: {expected_total_weight:.6f}")
    print(f"Actual aggregated: {result['aggregated_score']:.6f}")
    print(f"Actual total weight: {result['metadata']['total_weight']:.6f}")
    
    # Check that all are in Day 0
    if 0 in result['metadata']['sentiment_by_day']:
        day_0_data = result['metadata']['sentiment_by_day'][0]
        if day_0_data['count'] == 3:
            print("\n✓ PASS: All items correctly grouped as Day 0")
        else:
            print(f"\n✗ FAIL: Expected 3 items in Day 0, got {day_0_data['count']}")
    else:
        print("\n✗ FAIL: No Day 0 found!")
    
    if abs(result['aggregated_score'] - expected_aggregated) < 0.0001:
        print("✓ PASS: Aggregated score is correct")
    else:
        print(f"✗ FAIL: Aggregated score is incorrect!")
    
    return result


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Decay Calculation in News Signal Score")
    print("=" * 60)
    
    test_decay_calculation()
    test_decay_with_same_day_news()
    test_decay_with_old_news_excluded()
    test_decay_with_future_news_excluded()
    test_decay_progression()
    test_decay_with_datetime_objects()
    test_decay_edge_case_same_date_different_times()
    
    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

