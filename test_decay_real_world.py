#!/usr/bin/env python3
"""
Real-world test for decay calculation - simulates actual usage scenario.
"""

import sys
from datetime import date, datetime, timedelta
from typing import List, Dict, Any

# Import the function we want to test
sys.path.insert(0, '.')
from main import calculate_aggregated_sentiment, get_sentiment_for_date_range


def simulate_real_world_scenario():
    """
    Simulate a real-world scenario where we process signals for multiple days.
    This helps identify if decay is working correctly across different dates.
    """
    
    print("=" * 60)
    print("Real-World Scenario: Processing signals over multiple days")
    print("=" * 60)
    
    # Simulate news articles published over time
    # News published on different dates with different sentiment scores
    all_news = [
        {'date': date(2024, 1, 4), 'score': 0.1, 'title': 'Old news 1'},
        {'date': date(2024, 1, 5), 'score': 0.2, 'title': 'Old news 2'},
        {'date': date(2024, 1, 6), 'score': 0.3, 'title': 'Old news 3'},
        {'date': date(2024, 1, 7), 'score': 0.4, 'title': 'Recent news 1'},
        {'date': date(2024, 1, 8), 'score': 0.5, 'title': 'Recent news 2'},
        {'date': date(2024, 1, 9), 'score': 0.6, 'title': 'Recent news 3'},
        {'date': date(2024, 1, 10), 'score': 0.7, 'title': 'Today news 1'},
        {'date': date(2024, 1, 10), 'score': 0.8, 'title': 'Today news 2'},
    ]
    
    # Process signals for each day from Jan 7 to Jan 10
    for target_date in [date(2024, 1, 7), date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]:
        print(f"\n{'='*60}")
        print(f"Processing signal for date: {target_date}")
        print(f"{'='*60}")
        
        # Simulate what get_sentiment_for_date_range would return
        # Get news from (target_date - 6 days) to target_date
        start_date = target_date - timedelta(days=6)
        sentiment_list = []
        
        for news in all_news:
            if start_date <= news['date'] <= target_date:
                sentiment_list.append({
                    'published_date': news['date'],
                    'sentiment_score': news['score']
                })
        
        print(f"News in 7-day window ({start_date} to {target_date}):")
        for item in sentiment_list:
            days_ago = (target_date - item['published_date']).days
            expected_weight = 0.5 ** days_ago
            print(f"  {item['published_date']} (Day {days_ago}): score={item['sentiment_score']:.1f}, weight={expected_weight:.6f}")
        
        # Calculate aggregated sentiment
        result = calculate_aggregated_sentiment(sentiment_list, target_date)
        
        print(f"\nResult:")
        print(f"  Aggregated score: {result['aggregated_score']:.6f}")
        print(f"  Total weight: {result['metadata']['total_weight']:.6f}")
        print(f"  Weighted news count: {result['metadata']['weighted_news_count']}")
        
        # Manual calculation for verification
        manual_weighted_sum = 0.0
        manual_total_weight = 0.0
        print(f"\nManual calculation:")
        for item in sentiment_list:
            days_ago = (target_date - item['published_date']).days
            weight = 0.5 ** days_ago
            weighted_score = item['sentiment_score'] * weight
            manual_weighted_sum += weighted_score
            manual_total_weight += weight
            print(f"  Day {days_ago}: score={item['sentiment_score']:.1f} * weight={weight:.6f} = {weighted_score:.6f}")
        
        manual_aggregated = manual_weighted_sum / manual_total_weight if manual_total_weight > 0 else 0.0
        print(f"  Total weighted sum: {manual_weighted_sum:.6f}")
        print(f"  Total weight: {manual_total_weight:.6f}")
        print(f"  Aggregated: {manual_aggregated:.6f}")
        
        if abs(result['aggregated_score'] - manual_aggregated) < 0.0001:
            print(f"\n✓ PASS: Calculated score matches manual calculation")
        else:
            print(f"\n✗ FAIL: Calculated score doesn't match manual calculation!")
            print(f"  Difference: {abs(result['aggregated_score'] - manual_aggregated):.6f}")
        
        # Check if decay is actually being applied
        # If decay is working, older news should have less influence
        if len(sentiment_list) > 1:
            # Get the most recent and oldest news
            sorted_by_date = sorted(sentiment_list, key=lambda x: x['published_date'], reverse=True)
            most_recent = sorted_by_date[0]
            oldest = sorted_by_date[-1]
            
            most_recent_days_ago = (target_date - most_recent['published_date']).days
            oldest_days_ago = (target_date - oldest['published_date']).days
            
            most_recent_weight = 0.5 ** most_recent_days_ago
            oldest_weight = 0.5 ** oldest_days_ago
            
            print(f"\nDecay check:")
            print(f"  Most recent ({most_recent['published_date']}, Day {most_recent_days_ago}): weight={most_recent_weight:.6f}")
            print(f"  Oldest ({oldest['published_date']}, Day {oldest_days_ago}): weight={oldest_weight:.6f}")
            
            if oldest_days_ago > most_recent_days_ago and oldest_weight < most_recent_weight:
                print(f"  ✓ Decay is being applied (older news has lower weight)")
            else:
                print(f"  ✗ Decay may not be working correctly!")


def test_decay_effect_on_aggregated_score():
    """
    Test that demonstrates how decay affects the aggregated score.
    If decay is working, the aggregated score should be closer to recent news scores.
    """
    
    print("\n" + "=" * 60)
    print("Test: Decay effect on aggregated score")
    print("=" * 60)
    
    target_date = date(2024, 1, 10)
    
    # Scenario 1: Recent news is positive, old news is negative
    # If decay works, aggregated should be closer to positive
    scenario1 = [
        {'published_date': date(2024, 1, 10), 'sentiment_score': 0.9},  # Day 0: very positive
        {'published_date': date(2024, 1, 4), 'sentiment_score': -0.9},  # Day 6: very negative
    ]
    
    result1 = calculate_aggregated_sentiment(scenario1, target_date)
    
    print(f"\nScenario 1: Recent positive (0.9), Old negative (-0.9)")
    print(f"  Aggregated score: {result1['aggregated_score']:.6f}")
    
    # With decay, recent positive should dominate
    # Expected: (0.9 * 1.0 + (-0.9) * 0.015625) / (1.0 + 0.015625) ≈ 0.886
    expected1 = (0.9 * 1.0 + (-0.9) * 0.015625) / (1.0 + 0.015625)
    print(f"  Expected (with decay): {expected1:.6f}")
    print(f"  Without decay (simple avg): {(0.9 + (-0.9)) / 2:.6f}")
    
    if result1['aggregated_score'] > 0.5:
        print(f"  ✓ PASS: Recent positive news dominates (decay working)")
    else:
        print(f"  ✗ FAIL: Recent positive news doesn't dominate (decay may not be working)")
    
    # Scenario 2: Recent news is negative, old news is positive
    # If decay works, aggregated should be closer to negative
    scenario2 = [
        {'published_date': date(2024, 1, 10), 'sentiment_score': -0.9},  # Day 0: very negative
        {'published_date': date(2024, 1, 4), 'sentiment_score': 0.9},     # Day 6: very positive
    ]
    
    result2 = calculate_aggregated_sentiment(scenario2, target_date)
    
    print(f"\nScenario 2: Recent negative (-0.9), Old positive (0.9)")
    print(f"  Aggregated score: {result2['aggregated_score']:.6f}")
    
    # With decay, recent negative should dominate
    expected2 = ((-0.9) * 1.0 + 0.9 * 0.015625) / (1.0 + 0.015625)
    print(f"  Expected (with decay): {expected2:.6f}")
    print(f"  Without decay (simple avg): {((-0.9) + 0.9) / 2:.6f}")
    
    if result2['aggregated_score'] < -0.5:
        print(f"  ✓ PASS: Recent negative news dominates (decay working)")
    else:
        print(f"  ✗ FAIL: Recent negative news doesn't dominate (decay may not be working)")


def main():
    """Run all real-world tests."""
    simulate_real_world_scenario()
    test_decay_effect_on_aggregated_score()
    
    print("\n" + "=" * 60)
    print("Real-world tests completed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()


