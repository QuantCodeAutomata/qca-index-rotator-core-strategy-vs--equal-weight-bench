"""
Tests for momentum calculation module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.momentum import (
    calculate_momentum_12_1,
    calculate_momentum_generic,
    select_top_momentum_asset,
    get_momentum_scores,
    apply_cash_filter
)


def test_momentum_12_1_calculation():
    """Test 12-1 momentum calculation formula."""
    # Create test data
    dates = pd.date_range('2020-01-01', periods=20, freq='M')
    prices = pd.DataFrame({
        'Asset1': np.arange(100, 120),
        'Asset2': np.arange(200, 220)
    }, index=dates)
    
    # Calculate momentum
    momentum = calculate_momentum_12_1(prices)
    
    # First 13 rows should be NaN
    assert momentum.iloc[:13].isna().all().all()
    
    # Check formula: P_{t-1} / P_{t-13} - 1
    # At index 13: (prices[12] / prices[0]) - 1
    expected_asset1 = (prices['Asset1'].iloc[12] / prices['Asset1'].iloc[0]) - 1
    assert np.isclose(momentum['Asset1'].iloc[13], expected_asset1)
    
    print("✓ Test momentum 12-1 calculation passed")


def test_momentum_generic_lookback():
    """Test generic momentum calculation with different lookbacks."""
    dates = pd.date_range('2020-01-01', periods=15, freq='M')
    prices = pd.DataFrame({
        'Asset1': np.arange(100, 115)
    }, index=dates)
    
    # Test 3-month lookback
    momentum_3 = calculate_momentum_generic(prices, lookback=3)
    
    # First 4 rows should be NaN (lookback + 1)
    assert momentum_3.iloc[:4].isna().all().all()
    
    # Check formula at index 4: P_{t-1} / P_{t-4} - 1 = prices[3] / prices[0] - 1
    expected = (prices['Asset1'].iloc[3] / prices['Asset1'].iloc[0]) - 1
    assert np.isclose(momentum_3['Asset1'].iloc[4], expected)
    
    print("✓ Test momentum generic lookback passed")


def test_select_top_momentum():
    """Test selection of asset with highest momentum."""
    dates = pd.date_range('2020-01-01', periods=5, freq='M')
    momentum = pd.DataFrame({
        'Asset1': [0.05, 0.03, 0.10, 0.02, 0.08],
        'Asset2': [0.08, 0.07, 0.05, 0.09, 0.06],
        'Asset3': [0.03, 0.09, 0.08, 0.01, 0.10]
    }, index=dates)
    
    selected = select_top_momentum_asset(momentum, ['Asset1', 'Asset2', 'Asset3'])
    
    # Check each period
    assert selected.iloc[0] == 'Asset2'  # 0.08 is highest
    assert selected.iloc[1] == 'Asset3'  # 0.09 is highest
    assert selected.iloc[2] == 'Asset1'  # 0.10 is highest
    assert selected.iloc[3] == 'Asset2'  # 0.09 is highest
    assert selected.iloc[4] == 'Asset3'  # 0.10 is highest
    
    print("✓ Test select top momentum passed")


def test_get_momentum_scores():
    """Test extraction of momentum scores for selected assets."""
    dates = pd.date_range('2020-01-01', periods=3, freq='M')
    momentum = pd.DataFrame({
        'Asset1': [0.05, 0.03, 0.10],
        'Asset2': [0.08, 0.07, 0.05]
    }, index=dates)
    
    selected = pd.Series(['Asset2', 'Asset1', 'Asset2'], index=dates)
    
    scores = get_momentum_scores(momentum, selected)
    
    assert np.isclose(scores.iloc[0], 0.08)
    assert np.isclose(scores.iloc[1], 0.03)
    assert np.isclose(scores.iloc[2], 0.05)
    
    print("✓ Test get momentum scores passed")


def test_apply_cash_filter():
    """Test cash filter application."""
    dates = pd.date_range('2020-01-01', periods=5, freq='M')
    
    selected_equity = pd.Series(['Asset1', 'Asset2', 'Asset1', 'Asset2', 'Asset1'], index=dates)
    equity_momentum = pd.Series([0.10, -0.05, 0.08, 0.02, -0.03], index=dates)
    cash_momentum = pd.Series([0.01, -0.02, 0.10, -0.01, 0.05], index=dates)
    
    final = apply_cash_filter(selected_equity, equity_momentum, cash_momentum, 'SHY')
    
    # Month 0: equity (0.10) > cash (0.01) -> Asset1
    assert final.iloc[0] == 'Asset1'
    
    # Month 1: equity (-0.05) <= cash (-0.02) -> SHY
    assert final.iloc[1] == 'SHY'
    
    # Month 2: equity (0.08) <= cash (0.10) -> SHY
    assert final.iloc[2] == 'SHY'
    
    # Month 3: equity (0.02) > cash (-0.01) -> Asset2
    assert final.iloc[3] == 'Asset2'
    
    # Month 4: equity (-0.03) <= cash (0.05) -> SHY
    assert final.iloc[4] == 'SHY'
    
    print("✓ Test apply cash filter passed")


def test_momentum_with_edge_cases():
    """Test momentum calculation with edge cases."""
    # Test with single data point
    dates = pd.date_range('2020-01-01', periods=1, freq='M')
    prices = pd.DataFrame({'Asset1': [100]}, index=dates)
    
    momentum = calculate_momentum_12_1(prices)
    assert momentum.isna().all().all()
    
    # Test with negative returns
    dates = pd.date_range('2020-01-01', periods=20, freq='M')
    prices = pd.DataFrame({
        'Asset1': np.linspace(200, 100, 20)  # Declining prices
    }, index=dates)
    
    momentum = calculate_momentum_12_1(prices)
    
    # Momentum should be negative
    assert momentum['Asset1'].iloc[13] < 0
    
    print("✓ Test momentum edge cases passed")


def run_all_momentum_tests():
    """Run all momentum tests."""
    print("\n" + "=" * 80)
    print("RUNNING MOMENTUM TESTS")
    print("=" * 80)
    print()
    
    test_momentum_12_1_calculation()
    test_momentum_generic_lookback()
    test_select_top_momentum()
    test_get_momentum_scores()
    test_apply_cash_filter()
    test_momentum_with_edge_cases()
    
    print()
    print("=" * 80)
    print("ALL MOMENTUM TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_momentum_tests()
