"""
Statistical testing module.
Implements statistical tests for strategy evaluation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict
from src.performance import sharpe_ratio


def paired_t_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Tuple[float, float]:
    """
    Perform paired t-test on return differences.
    
    Tests whether the mean return difference is significantly different from zero.
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns
    
    Returns:
    --------
    Tuple[float, float]
        (t-statistic, two-sided p-value)
    """
    # Align series
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned) < 2:
        return np.nan, np.nan
    
    # Calculate differences
    differences = aligned['strategy'] - aligned['benchmark']
    
    # Perform t-test
    t_stat = differences.mean() / (differences.std() / np.sqrt(len(differences)))
    
    # Calculate p-value (two-sided)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(differences) - 1))
    
    return t_stat, p_value


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    random_seed: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Calculate bootstrap confidence interval for Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of periodic returns
    n_bootstrap : int
        Number of bootstrap samples (default 10,000)
    confidence_level : float
        Confidence level (default 0.95 for 95% CI)
    risk_free_rate : float
        Annualized risk-free rate
    periods_per_year : int
        Number of periods per year (12 for monthly)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    Tuple[float, float, np.ndarray]
        (lower bound, upper bound, array of bootstrap Sharpe ratios)
    """
    # Remove NaN values
    returns_clean = returns.dropna().values
    n = len(returns_clean)
    
    if n < 2:
        return np.nan, np.nan, np.array([])
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Bootstrap
    bootstrap_sharpes = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(returns_clean, size=n, replace=True)
        sample_series = pd.Series(sample)
        
        # Calculate Sharpe ratio for this sample
        bootstrap_sharpes[i] = sharpe_ratio(sample_series, risk_free_rate, periods_per_year)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_sharpes, lower_percentile)
    upper_bound = np.percentile(bootstrap_sharpes, upper_percentile)
    
    return lower_bound, upper_bound, bootstrap_sharpes


def calculate_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 12
) -> float:
    """
    Calculate Information Ratio (annualized).
    
    Formula: IR = mean(active_returns) / std(active_returns) * sqrt(periods_per_year)
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns
    periods_per_year : int
        Number of periods per year (12 for monthly)
    
    Returns:
    --------
    float
        Information ratio
    """
    # Align series
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned) < 2:
        return np.nan
    
    # Calculate active returns
    active_returns = aligned['strategy'] - aligned['benchmark']
    
    # Calculate Information Ratio
    ir = (active_returns.mean() / active_returns.std()) * np.sqrt(periods_per_year)
    
    return ir


def calculate_tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 12
) -> float:
    """
    Calculate tracking error (annualized).
    
    Formula: TE = std(active_returns) * sqrt(periods_per_year)
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns
    periods_per_year : int
        Number of periods per year (12 for monthly)
    
    Returns:
    --------
    float
        Tracking error
    """
    # Align series
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned) < 2:
        return np.nan
    
    # Calculate active returns
    active_returns = aligned['strategy'] - aligned['benchmark']
    
    # Calculate tracking error
    te = active_returns.std() * np.sqrt(periods_per_year)
    
    return te


def statistical_summary(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, any]:
    """
    Generate comprehensive statistical summary.
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level for CI
    random_seed : int
        Random seed
    
    Returns:
    --------
    Dict[str, any]
        Dictionary with statistical test results
    """
    # Paired t-test
    t_stat, p_value = paired_t_test(strategy_returns, benchmark_returns)
    
    # Bootstrap CI for strategy
    strategy_ci_lower, strategy_ci_upper, strategy_boot = bootstrap_sharpe_ci(
        strategy_returns,
        n_bootstrap,
        confidence_level,
        random_seed=random_seed
    )
    
    # Bootstrap CI for benchmark
    bench_ci_lower, bench_ci_upper, bench_boot = bootstrap_sharpe_ci(
        benchmark_returns,
        n_bootstrap,
        confidence_level,
        random_seed=random_seed
    )
    
    # Information ratio
    ir = calculate_information_ratio(strategy_returns, benchmark_returns)
    
    # Tracking error
    te = calculate_tracking_error(strategy_returns, benchmark_returns)
    
    summary = {
        'Paired t-test': {
            't-statistic': t_stat,
            'p-value': p_value,
            'significant (p<0.05)': p_value < 0.05 if not np.isnan(p_value) else False
        },
        'Bootstrap Sharpe CI (Strategy)': {
            'lower_bound': strategy_ci_lower,
            'upper_bound': strategy_ci_upper
        },
        'Bootstrap Sharpe CI (Benchmark)': {
            'lower_bound': bench_ci_lower,
            'upper_bound': bench_ci_upper
        },
        'Information Ratio': ir,
        'Tracking Error': te
    }
    
    return summary
