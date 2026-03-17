"""
Momentum signal calculation module.
Implements various momentum calculations with different lookback periods.
"""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 12-1 momentum signal (11-month return with 1-month skip).
    
    Formula: Mom_{i,t} = P_{i,t-1} / P_{i,t-13} - 1
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with date index and price columns for each asset
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with momentum scores for each asset
        First 13 rows will be NaN due to lookback requirement
    """
    # Calculate momentum: P_{t-1} / P_{t-13} - 1
    momentum = prices.shift(1) / prices.shift(13) - 1
    return momentum


def calculate_momentum_generic(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Calculate momentum signal with generic lookback period and 1-month skip.
    
    Formula: Mom_{i,t}^{(L)} = P_{i,t-1} / P_{i,t-(L+1)} - 1
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with date index and price columns for each asset
    lookback : int
        Lookback period in months (e.g., 3, 6, 9, 12)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with momentum scores for each asset
        First (lookback+1) rows will be NaN
    """
    # Calculate momentum: P_{t-1} / P_{t-(lookback+1)} - 1
    momentum = prices.shift(1) / prices.shift(lookback + 1) - 1
    return momentum


def select_top_momentum_asset(
    momentum: pd.DataFrame,
    asset_universe: list
) -> pd.Series:
    """
    Select the asset with the highest momentum score from a given universe.
    
    Parameters:
    -----------
    momentum : pd.DataFrame
        DataFrame with momentum scores
    asset_universe : list
        List of asset tickers to consider
    
    Returns:
    --------
    pd.Series
        Series with the selected asset ticker for each date
    """
    # Filter to asset universe
    momentum_subset = momentum[asset_universe]
    
    # Select asset with highest momentum
    selected = momentum_subset.idxmax(axis=1)
    
    return selected


def get_momentum_scores(
    momentum: pd.DataFrame,
    selected_assets: pd.Series
) -> pd.Series:
    """
    Extract momentum scores for selected assets.
    
    Parameters:
    -----------
    momentum : pd.DataFrame
        DataFrame with momentum scores for all assets
    selected_assets : pd.Series
        Series with selected asset ticker for each date
    
    Returns:
    --------
    pd.Series
        Series with momentum score for the selected asset at each date
    """
    # Extract momentum score for selected asset at each date
    scores = pd.Series(index=selected_assets.index, dtype=float)
    for date, asset in selected_assets.items():
        if pd.notna(asset):
            scores.loc[date] = momentum.loc[date, asset]
    
    return scores


def apply_cash_filter(
    selected_equity: pd.Series,
    equity_momentum: pd.Series,
    cash_momentum: pd.Series,
    cash_ticker: str = 'SHY'
) -> pd.Series:
    """
    Apply cash filter: switch to cash if selected equity momentum <= cash momentum.
    
    Parameters:
    -----------
    selected_equity : pd.Series
        Series with selected equity ticker for each date
    equity_momentum : pd.Series
        Series with momentum score of selected equity
    cash_momentum : pd.Series
        Series with momentum score of cash asset (SHY)
    cash_ticker : str
        Ticker symbol for cash asset (default 'SHY')
    
    Returns:
    --------
    pd.Series
        Series with final asset selection (equity or cash)
    """
    # Create final selection series
    final_selection = selected_equity.copy()
    
    # Apply filter: if equity momentum <= cash momentum, hold cash
    cash_filter_triggered = equity_momentum <= cash_momentum
    final_selection[cash_filter_triggered] = cash_ticker
    
    return final_selection


def calculate_all_lookback_momentum(
    prices: pd.DataFrame,
    lookback_periods: list = [3, 6, 9, 12]
) -> Dict[int, pd.DataFrame]:
    """
    Calculate momentum signals for multiple lookback periods.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with date index and price columns
    lookback_periods : list
        List of lookback periods to calculate
    
    Returns:
    --------
    Dict[int, pd.DataFrame]
        Dictionary mapping lookback period to momentum DataFrame
    """
    momentum_dict = {}
    
    for lookback in lookback_periods:
        momentum_dict[lookback] = calculate_momentum_generic(prices, lookback)
    
    return momentum_dict
