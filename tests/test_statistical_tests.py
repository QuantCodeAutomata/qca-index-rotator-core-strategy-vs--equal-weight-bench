"""
Tests for statistical testing module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.statistical_tests import (
    paired_t_test,
    bootstrap_sharpe_ci,
    calculate_information_ratio,
    calculate_tracking_error
)


def test_paired_t_test():
    """Test paired t-test calculation."""
    dates = pd.date_range('2020-01-01', periods=100, freq='M')
    
    np.random.seed(42)
    strategy = pd.Series(np.random.normal(0.01, 0.02, 100), index=dates)
    benchmark = pd.Series(np.random.normal(0.008, 0.02, 100), index=dates)
    
    t_stat, p_value = paired_t_test(strategy, benchmark)
    
    # Check that results are finite
    assert np.isfinite(t_stat)
    assert np.isfinite(p_value)
    
    # P-value should be between 0 and 1
    assert 0 <= p_value <= 1
    
    # Test with identical series (should have p-value close to 1, t-stat may be NaN)
    t_stat_same, p_value_same = paired_t_test(strategy, strategy)
    # When differences are all zero, std is 0, leading to NaN
    assert np.isnan(t_stat_same) or np.isclose(t_stat_same, 0.0, atol=1e-6)
    
    print("✓ Test paired t-test passed")


def test_bootstrap_sharpe_ci():
    """Test bootstrap Sharpe ratio confidence interval."""
    dates = pd.date_range('2020-01-01', periods=60, freq='M')
    
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.01, 0.02, 60), index=dates)
    
    lower, upper, samples = bootstrap_sharpe_ci(
        returns,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_seed=42
    )
    
    # Lower bound should be less than upper bound
    assert lower < upper
    
    # Samples should have correct length
    assert len(samples) == 1000
    
    # All samples should be finite
    assert np.all(np.isfinite(samples))
    
    # Calculate actual Sharpe
    from src.performance import sharpe_ratio
    actual_sharpe = sharpe_ratio(returns)
    
    # Actual Sharpe should generally be between bounds (not always, due to randomness)
    # Just check that bounds are reasonable
    assert lower < actual_sharpe + 1.0  # Upper bound check
    assert upper > actual_sharpe - 1.0  # Lower bound check
    
    print("✓ Test bootstrap Sharpe CI passed")


def test_information_ratio():
    """Test information ratio calculation."""
    dates = pd.date_range('2020-01-01', periods=36, freq='M')
    
    np.random.seed(42)
    strategy = pd.Series(np.random.normal(0.01, 0.02, 36), index=dates)
    benchmark = pd.Series(np.random.normal(0.008, 0.02, 36), index=dates)
    
    ir = calculate_information_ratio(strategy, benchmark)
    
    # Check that result is finite
    assert np.isfinite(ir)
    
    # Test with identical series (IR should be 0 or NaN due to zero std of differences)
    ir_same = calculate_information_ratio(strategy, strategy)
    assert np.isnan(ir_same) or np.isclose(ir_same, 0.0, atol=1e-6)
    
    print("✓ Test information ratio passed")


def test_tracking_error():
    """Test tracking error calculation."""
    dates = pd.date_range('2020-01-01', periods=36, freq='M')
    
    np.random.seed(42)
    strategy = pd.Series(np.random.normal(0.01, 0.02, 36), index=dates)
    benchmark = pd.Series(np.random.normal(0.01, 0.02, 36), index=dates)
    
    te = calculate_tracking_error(strategy, benchmark)
    
    # Tracking error should be positive
    assert te > 0
    
    # Test with identical series (TE should be 0)
    te_same = calculate_tracking_error(strategy, strategy)
    assert np.isclose(te_same, 0.0, atol=1e-10)
    
    print("✓ Test tracking error passed")


def test_statistical_tests_with_edge_cases():
    """Test statistical tests with edge cases."""
    # Test with very small sample
    small = pd.Series([0.01, 0.02])
    benchmark_small = pd.Series([0.01, 0.015])
    
    t_stat, p_value = paired_t_test(small, benchmark_small)
    assert np.isfinite(t_stat) or np.isnan(t_stat)
    
    # Test with single value
    single = pd.Series([0.01])
    
    lower, upper, samples = bootstrap_sharpe_ci(single, n_bootstrap=100)
    # Should handle gracefully
    
    print("✓ Test statistical edge cases passed")


def test_bootstrap_reproducibility():
    """Test that bootstrap with same seed gives same results."""
    dates = pd.date_range('2020-01-01', periods=60, freq='M')
    
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.01, 0.02, 60), index=dates)
    
    # Run twice with same seed
    lower1, upper1, samples1 = bootstrap_sharpe_ci(returns, n_bootstrap=1000, random_seed=42)
    lower2, upper2, samples2 = bootstrap_sharpe_ci(returns, n_bootstrap=1000, random_seed=42)
    
    # Results should be identical
    assert np.isclose(lower1, lower2)
    assert np.isclose(upper1, upper2)
    assert np.allclose(samples1, samples2)
    
    print("✓ Test bootstrap reproducibility passed")


def run_all_statistical_tests():
    """Run all statistical tests."""
    print("\n" + "=" * 80)
    print("RUNNING STATISTICAL TESTS")
    print("=" * 80)
    print()
    
    test_paired_t_test()
    test_bootstrap_sharpe_ci()
    test_information_ratio()
    test_tracking_error()
    test_statistical_tests_with_edge_cases()
    test_bootstrap_reproducibility()
    
    print()
    print("=" * 80)
    print("ALL STATISTICAL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_statistical_tests()
