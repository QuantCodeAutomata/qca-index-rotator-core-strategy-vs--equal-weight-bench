"""
Strategy implementation module.
Contains implementations of Index Rotator and Equal-Weight Benchmark strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from src.momentum import (
    calculate_momentum_12_1,
    calculate_momentum_generic,
    select_top_momentum_asset,
    get_momentum_scores,
    apply_cash_filter
)


def calculate_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns from prices.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with date index and price columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with monthly returns
    """
    returns = prices.pct_change()
    return returns


def apply_transaction_costs(
    returns: pd.Series,
    asset_changes: pd.Series,
    round_trip_cost: float = 0.001
) -> pd.Series:
    """
    Apply transaction costs to returns when assets change.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of monthly returns
    asset_changes : pd.Series
        Boolean series indicating when asset changes
    round_trip_cost : float
        Round-trip transaction cost (default 0.001 = 10 bps)
    
    Returns:
    --------
    pd.Series
        Returns after transaction costs
    """
    # Deduct transaction cost when asset changes
    returns_after_tc = returns.copy()
    returns_after_tc[asset_changes] = returns_after_tc[asset_changes] - round_trip_cost
    
    return returns_after_tc


def index_rotator_strategy(
    prices: pd.DataFrame,
    equity_tickers: list = ['SPY', 'QQQ', 'IWM', 'DIA'],
    cash_ticker: str = 'SHY',
    apply_filter: bool = True,
    lookback: int = 12,
    transaction_cost: float = 0.001
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Implement the Index Rotator strategy.
    
    Strategy logic:
    1. Calculate momentum for all assets (12-1 momentum by default)
    2. Select equity with highest momentum
    3. Apply cash filter if enabled: switch to cash if equity momentum <= cash momentum
    4. Apply transaction costs when asset changes
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with monthly prices
    equity_tickers : list
        List of equity ETF tickers
    cash_ticker : str
        Cash ETF ticker (default 'SHY')
    apply_filter : bool
        Whether to apply cash filter (default True)
    lookback : int
        Momentum lookback period (default 12)
    transaction_cost : float
        Round-trip transaction cost (default 0.001 = 10 bps)
    
    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.DataFrame]
        (monthly returns after costs, selected assets, momentum scores)
    """
    # Calculate momentum
    momentum = calculate_momentum_generic(prices, lookback)
    
    # Select top momentum equity
    selected_equity = select_top_momentum_asset(momentum, equity_tickers)
    equity_momentum = get_momentum_scores(momentum, selected_equity)
    
    # Apply cash filter if enabled
    if apply_filter:
        cash_momentum = momentum[cash_ticker]
        selected_asset = apply_cash_filter(
            selected_equity,
            equity_momentum,
            cash_momentum,
            cash_ticker
        )
    else:
        selected_asset = selected_equity
    
    # Calculate returns
    all_returns = calculate_monthly_returns(prices)
    
    # Extract returns for selected assets
    strategy_returns = pd.Series(index=selected_asset.index, dtype=float)
    for date, asset in selected_asset.items():
        if pd.notna(asset) and date in all_returns.index:
            strategy_returns.loc[date] = all_returns.loc[date, asset]
    
    # Identify asset changes
    asset_changes = selected_asset != selected_asset.shift(1)
    # Don't charge transaction cost on the first trade
    asset_changes.iloc[0] = False
    
    # Apply transaction costs
    strategy_returns = apply_transaction_costs(
        strategy_returns,
        asset_changes,
        transaction_cost
    )
    
    return strategy_returns, selected_asset, momentum


def equal_weight_benchmark(
    prices: pd.DataFrame,
    tickers: list = ['SPY', 'QQQ', 'IWM', 'DIA'],
    transaction_cost_bps: float = 0.0005
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Implement equal-weight benchmark strategy with monthly rebalancing.
    
    Strategy logic:
    1. Allocate 25% (equal weight) to each of 4 equity ETFs
    2. Rebalance monthly back to equal weights
    3. Apply transaction costs based on weight drift
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with monthly prices
    tickers : list
        List of ETF tickers for benchmark
    transaction_cost_bps : float
        One-way transaction cost (default 0.0005 = 5 bps)
    
    Returns:
    --------
    Tuple[pd.Series, pd.DataFrame]
        (monthly returns after costs, weight history)
    """
    # Calculate returns
    all_returns = calculate_monthly_returns(prices[tickers])
    
    # Initialize weights
    n_assets = len(tickers)
    target_weight = 1.0 / n_assets
    
    # Track weights and returns
    weights_history = pd.DataFrame(index=all_returns.index, columns=tickers)
    benchmark_returns = pd.Series(index=all_returns.index, dtype=float)
    
    # Initialize weights
    weights_before = pd.Series(target_weight, index=tickers)
    
    for i, date in enumerate(all_returns.index):
        if pd.isna(all_returns.loc[date]).all():
            continue
        
        # For the first month
        if i == 0:
            monthly_return = (weights_before * all_returns.loc[date]).sum()
            benchmark_returns.loc[date] = monthly_return
            weights_history.loc[date] = weights_before
        else:
            # Get current month's returns
            curr_returns = all_returns.loc[date]
            
            # Calculate weights after previous month's returns (before rebalancing)
            weights_after_returns = weights_before * (1 + curr_returns)
            weights_after_returns = weights_after_returns / weights_after_returns.sum()
            
            # Calculate transaction costs
            weight_drift = (weights_after_returns - target_weight).abs().sum()
            tc = transaction_cost_bps * weight_drift
            
            # Monthly return before transaction costs
            monthly_return = (weights_before * curr_returns).sum()
            
            # Apply transaction costs
            benchmark_returns.loc[date] = monthly_return - tc
            
            # Record weights before this month (used for return calculation)
            weights_history.loc[date] = weights_before
            
            # Rebalance to target weights for next month
            weights_before = pd.Series(target_weight, index=tickers)
    
    return benchmark_returns, weights_history


def calculate_turnover(selected_assets: pd.Series) -> float:
    """
    Calculate average monthly turnover for a strategy.
    
    For Index Rotator: % of months where asset changes
    
    Parameters:
    -----------
    selected_assets : pd.Series
        Series of selected assets over time
    
    Returns:
    --------
    float
        Average monthly turnover as percentage
    """
    # Count asset changes
    asset_changes = (selected_assets != selected_assets.shift(1)).sum()
    
    # Exclude first month (not a real change)
    asset_changes = asset_changes - 1
    
    # Total months (exclude first month)
    total_months = len(selected_assets) - 1
    
    # Turnover = changes / total months * 100%
    turnover = (asset_changes / total_months) * 100 if total_months > 0 else 0
    
    return turnover


def calculate_benchmark_turnover(weights_history: pd.DataFrame, target_weight: float = 0.25) -> float:
    """
    Calculate average monthly turnover for equal-weight benchmark.
    
    Parameters:
    -----------
    weights_history : pd.DataFrame
        DataFrame with weight history
    target_weight : float
        Target weight for each asset
    
    Returns:
    --------
    float
        Average monthly turnover as percentage
    """
    # Calculate weight drifts (this would require returns, so approximate)
    # For equal-weight benchmark, use a simplified calculation
    # Turnover = average sum of absolute weight adjustments
    
    # Since we rebalance to target every month, turnover depends on drift
    # This is an approximation - actual calculation requires return data
    # For now, return a placeholder
    
    return 0.0  # Will be calculated properly in the experiment scripts


def get_months_in_cash(selected_assets: pd.Series, cash_ticker: str = 'SHY') -> Tuple[int, list]:
    """
    Count and list months when the strategy held cash.
    
    Parameters:
    -----------
    selected_assets : pd.Series
        Series of selected assets
    cash_ticker : str
        Cash asset ticker
    
    Returns:
    --------
    Tuple[int, list]
        (count of cash months, list of dates in cash)
    """
    cash_months = selected_assets[selected_assets == cash_ticker]
    return len(cash_months), cash_months.index.tolist()
