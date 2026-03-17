"""
Tests for performance metrics calculation module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.performance import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    maximum_drawdown,
    calmar_ratio,
    hit_rate
)


def test_annualized_return():
    """Test annualized return calculation."""
    # Test with constant positive returns
    dates = pd.date_range('2020-01-01', periods=12, freq='M')
    returns = pd.Series([0.01] * 12, index=dates)  # 1% per month
    
    ann_ret = annualized_return(returns, periods_per_year=12)
    
    # Expected: (1.01)^12 - 1 ≈ 0.1268
    expected = (1.01 ** 12) - 1
    assert np.isclose(ann_ret, expected, rtol=1e-6)
    
    # Test with 24 months
    returns_24 = pd.Series([0.01] * 24, index=pd.date_range('2020-01-01', periods=24, freq='M'))
    ann_ret_24 = annualized_return(returns_24, periods_per_year=12)
    
    # Should still be approximately same annualized return
    assert np.isclose(ann_ret_24, expected, rtol=1e-6)
    
    print("✓ Test annualized return passed")


def test_annualized_volatility():
    """Test annualized volatility calculation."""
    # Test with known volatility
    dates = pd.date_range('2020-01-01', periods=12, freq='M')
    
    # Create returns with std = 0.02
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 12), index=dates)
    
    ann_vol = annualized_volatility(returns, periods_per_year=12)
    
    # Expected: std * sqrt(12)
    expected = returns.std() * np.sqrt(12)
    assert np.isclose(ann_vol, expected)
    
    print("✓ Test annualized volatility passed")


def test_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    dates = pd.date_range('2020-01-01', periods=12, freq='M')
    returns = pd.Series([0.01] * 12, index=dates)
    
    # Test with zero risk-free rate
    sr = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=12)
    
    ann_ret = annualized_return(returns, 12)
    ann_vol = annualized_volatility(returns, 12)
    expected = ann_ret / ann_vol
    
    assert np.isclose(sr, expected)
    
    # Test with non-zero risk-free rate
    sr_rf = sharpe_ratio(returns, risk_free_rate=0.03, periods_per_year=12)
    expected_rf = (ann_ret - 0.03) / ann_vol
    
    assert np.isclose(sr_rf, expected_rf)
    
    print("✓ Test Sharpe ratio passed")


def test_maximum_drawdown():
    """Test maximum drawdown calculation."""
    # Test with known drawdown
    dates = pd.date_range('2020-01-01', periods=10, freq='M')
    
    # Create scenario: up 10%, up 10%, down 20%, down 10%, up 5%
    returns = pd.Series([0.10, 0.10, -0.20, -0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], index=dates)
    
    mdd = maximum_drawdown(returns)
    
    # Calculate cumulative wealth
    wealth = (1 + returns).cumprod()
    # Start: 1.0
    # After month 1: 1.10
    # After month 2: 1.21 (peak)
    # After month 3: 0.968
    # After month 4: 0.8712
    # Drawdown from peak: (1.21 - 0.8712) / 1.21 = 0.28
    
    assert mdd > 0.27 and mdd < 0.29
    
    # Test with no drawdown (only gains)
    returns_up = pd.Series([0.01] * 10, index=dates)
    mdd_up = maximum_drawdown(returns_up)
    assert np.isclose(mdd_up, 0.0)
    
    print("✓ Test maximum drawdown passed")


def test_calmar_ratio():
    """Test Calmar ratio calculation."""
    dates = pd.date_range('2020-01-01', periods=12, freq='M')
    returns = pd.Series([0.02, 0.01, -0.05, 0.03, 0.02, 0.01, 0.02, 0.03, 0.01, 0.02, 0.01, 0.02], index=dates)
    
    calmar = calmar_ratio(returns, periods_per_year=12)
    
    # Calculate expected
    ann_ret = annualized_return(returns, 12)
    mdd = maximum_drawdown(returns)
    expected = ann_ret / mdd
    
    assert np.isclose(calmar, expected)
    
    print("✓ Test Calmar ratio passed")


def test_hit_rate():
    """Test hit rate calculation."""
    dates = pd.date_range('2020-01-01', periods=10, freq='M')
    
    strategy = pd.Series([0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.01, 0.03], index=dates)
    benchmark = pd.Series([0.01, 0.01, 0.02, 0.02, -0.01, 0.02, 0.03, 0.01, 0.01, 0.02], index=dates)
    
    hr = hit_rate(strategy, benchmark)
    
    # Count wins: month 0 (0.02>0.01), month 2 (0.03>0.02), month 4 (-0.02<-0.01), 
    # month 5 (0.04>0.02), month 9 (0.03>0.02)
    # Wins: 0, 2, 5, 9 = 4 out of 10 = 40%
    
    wins = (strategy > benchmark).sum()
    expected = (wins / len(strategy)) * 100
    
    assert np.isclose(hr, expected)
    
    print("✓ Test hit rate passed")


def test_performance_with_edge_cases():
    """Test performance metrics with edge cases."""
    # Test with empty series
    empty = pd.Series(dtype=float)
    assert np.isnan(annualized_return(empty))
    assert np.isnan(annualized_volatility(empty))
    
    # Test with single value
    single = pd.Series([0.01])
    assert not np.isnan(annualized_return(single))
    
    # Test with zero volatility
    zero_vol = pd.Series([0.0] * 10)
    vol = annualized_volatility(zero_vol)
    assert np.isclose(vol, 0.0)
    
    # Sharpe should be NaN with zero volatility
    sr = sharpe_ratio(zero_vol)
    assert np.isnan(sr)
    
    print("✓ Test performance edge cases passed")


def test_mathematical_properties():
    """Test mathematical properties of metrics."""
    dates = pd.date_range('2020-01-01', periods=24, freq='M')
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.005, 0.02, 24), index=dates)
    
    # Maximum drawdown should be between 0 and 1
    mdd = maximum_drawdown(returns)
    assert 0 <= mdd <= 1
    
    # Annualized volatility should be positive
    vol = annualized_volatility(returns)
    assert vol >= 0
    
    # Hit rate should be between 0 and 100
    benchmark = pd.Series(np.random.normal(0.004, 0.02, 24), index=dates)
    hr = hit_rate(returns, benchmark)
    assert 0 <= hr <= 100
    
    print("✓ Test mathematical properties passed")


def run_all_performance_tests():
    """Run all performance tests."""
    print("\n" + "=" * 80)
    print("RUNNING PERFORMANCE TESTS")
    print("=" * 80)
    print()
    
    test_annualized_return()
    test_annualized_volatility()
    test_sharpe_ratio()
    test_maximum_drawdown()
    test_calmar_ratio()
    test_hit_rate()
    test_performance_with_edge_cases()
    test_mathematical_properties()
    
    print()
    print("=" * 80)
    print("ALL PERFORMANCE TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_performance_tests()
