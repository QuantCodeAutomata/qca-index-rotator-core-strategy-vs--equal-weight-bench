"""
Performance metrics calculation module.
Implements standard financial performance metrics for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Calculate annualized return from monthly returns.
    
    Formula: R_ann = (product(1+r_t))^{periods_per_year/T} - 1
    
    Parameters:
    -----------
    returns : pd.Series
        Series of periodic returns
    periods_per_year : int
        Number of periods per year (12 for monthly)
    
    Returns:
    --------
    float
        Annualized return
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate cumulative return
    cumulative_return = (1 + returns_clean).prod()
    
    # Annualize
    n_periods = len(returns_clean)
    ann_return = cumulative_return ** (periods_per_year / n_periods) - 1
    
    return ann_return


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Calculate annualized volatility from monthly returns.
    
    Formula: sigma_ann = std(r_t) * sqrt(periods_per_year)
    
    Parameters:
    -----------
    returns : pd.Series
        Series of periodic returns
    periods_per_year : int
        Number of periods per year (12 for monthly)
    
    Returns:
    --------
    float
        Annualized volatility
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate standard deviation
    std_dev = returns_clean.std()
    
    # Annualize
    ann_vol = std_dev * np.sqrt(periods_per_year)
    
    return ann_vol


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12
) -> float:
    """
    Calculate Sharpe ratio.
    
    Formula: SR = (R_ann - rf) / sigma_ann
    
    Parameters:
    -----------
    returns : pd.Series
        Series of periodic returns
    risk_free_rate : float
        Annualized risk-free rate (default 0.0)
    periods_per_year : int
        Number of periods per year (12 for monthly)
    
    Returns:
    --------
    float
        Sharpe ratio
    """
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    
    sharpe = (ann_ret - risk_free_rate) / ann_vol
    
    return sharpe


def maximum_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Formula: MDD = max_t[(max_{s<=t}V_s - V_t) / max_{s<=t}V_s]
    
    Parameters:
    -----------
    returns : pd.Series
        Series of periodic returns
    
    Returns:
    --------
    float
        Maximum drawdown (positive value)
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate cumulative returns
    cumulative = (1 + returns_clean).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown (as positive value)
    mdd = abs(drawdown.min())
    
    return mdd


def calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Calculate Calmar ratio.
    
    Formula: Calmar = R_ann / |MDD|
    
    Parameters:
    -----------
    returns : pd.Series
        Series of periodic returns
    periods_per_year : int
        Number of periods per year (12 for monthly)
    
    Returns:
    --------
    float
        Calmar ratio
    """
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = maximum_drawdown(returns)
    
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    
    calmar = ann_ret / mdd
    
    return calmar


def hit_rate(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate hit rate: percentage of periods where strategy outperforms benchmark.
    
    Parameters:
    -----------
    strategy_returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns
    
    Returns:
    --------
    float
        Hit rate as percentage (0-100)
    """
    # Align series
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned) == 0:
        return np.nan
    
    # Count periods where strategy > benchmark
    wins = (aligned['strategy'] > aligned['benchmark']).sum()
    
    # Calculate hit rate
    hit_rate_pct = (wins / len(aligned)) * 100
    
    return hit_rate_pct


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series over time.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of periodic returns
    
    Returns:
    --------
    pd.Series
        Series of drawdowns at each point in time
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return pd.Series(dtype=float)
    
    # Calculate cumulative returns
    cumulative = (1 + returns_clean).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown


def calculate_cumulative_returns(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """
    Calculate cumulative returns (wealth index).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of periodic returns
    start_value : float
        Starting value (default 1.0)
    
    Returns:
    --------
    pd.Series
        Series of cumulative wealth
    """
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return pd.Series(dtype=float)
    
    # Calculate cumulative returns
    cumulative = start_value * (1 + returns_clean).cumprod()
    
    return cumulative


def calculate_all_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate all performance metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns (optional, for hit rate)
    periods_per_year : int
        Number of periods per year (12 for monthly)
    risk_free_rate : float
        Annualized risk-free rate
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of all performance metrics
    """
    metrics = {
        'Annualized Return': annualized_return(returns, periods_per_year),
        'Annualized Volatility': annualized_volatility(returns, periods_per_year),
        'Sharpe Ratio': sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'Maximum Drawdown': maximum_drawdown(returns),
        'Calmar Ratio': calmar_ratio(returns, periods_per_year),
    }
    
    if benchmark_returns is not None:
        metrics['Hit Rate (%)'] = hit_rate(returns, benchmark_returns)
    
    return metrics


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Format multiple strategies' metrics into a comparison table.
    
    Parameters:
    -----------
    metrics_dict : Dict[str, Dict[str, float]]
        Dictionary mapping strategy names to their metrics
    
    Returns:
    --------
    pd.DataFrame
        Formatted comparison table
    """
    df = pd.DataFrame(metrics_dict).T
    return df
