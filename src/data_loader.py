"""
Data loader module for downloading and preprocessing ETF price data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple
from datetime import datetime


def download_etf_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1mo'
) -> pd.DataFrame:
    """
    Download adjusted closing prices for ETFs from Yahoo Finance.
    
    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols to download
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    interval : str
        Data interval ('1mo' for monthly, '1d' for daily)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with date index and columns for each ticker's adjusted close price
    """
    # Download data
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    
    # Extract closing prices
    if len(tickers) == 1:
        prices = pd.DataFrame(data['Close'])
        prices.columns = tickers
    else:
        prices = data['Close']
    
    # Ensure we have all tickers
    for ticker in tickers:
        if ticker not in prices.columns:
            raise ValueError(f"Ticker {ticker} not found in downloaded data")
    
    # Remove any rows with NaN values
    prices = prices.dropna()
    
    return prices


def resample_to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to month-end prices.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with date index and price columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame resampled to month-end frequency
    """
    # Resample to month-end, taking the last value of each month
    month_end_prices = prices.resample('M').last()
    return month_end_prices


def validate_data(
    prices: pd.DataFrame,
    required_tickers: List[str],
    min_periods: int = 14
) -> Tuple[bool, str]:
    """
    Validate that the price data meets requirements.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with date index and price columns
    required_tickers : List[str]
        List of required ticker symbols
    min_periods : int
        Minimum number of periods required (default 14 for 12-1 momentum + 1 month)
    
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Check all required tickers are present
    missing_tickers = set(required_tickers) - set(prices.columns)
    if missing_tickers:
        return False, f"Missing tickers: {missing_tickers}"
    
    # Check minimum number of periods
    if len(prices) < min_periods:
        return False, f"Insufficient data: {len(prices)} periods, need at least {min_periods}"
    
    # Check for NaN values
    if prices.isnull().any().any():
        return False, "Data contains NaN values"
    
    # Check for non-positive prices
    if (prices <= 0).any().any():
        return False, "Data contains non-positive prices"
    
    return True, ""


def load_strategy_data(
    start_date: str = '2003-01-01',
    end_date: str = '2023-12-31'
) -> pd.DataFrame:
    """
    Load all required data for the Index Rotator strategy.
    
    Parameters:
    -----------
    start_date : str
        Start date for data download (default '2003-01-01' to have history for Jan 2004 signals)
    end_date : str
        End date for data download
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with monthly prices for SPY, QQQ, IWM, DIA, SHY
    """
    tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'SHY']
    
    # Download data
    prices = download_etf_data(tickers, start_date, end_date, interval='1mo')
    
    # Validate data
    is_valid, error_msg = validate_data(prices, tickers)
    if not is_valid:
        raise ValueError(f"Data validation failed: {error_msg}")
    
    return prices


def get_price_history(
    prices: pd.DataFrame,
    start_month: str,
    end_month: str
) -> pd.DataFrame:
    """
    Extract price history for a specific date range.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Full price DataFrame
    start_month : str
        Start month in format 'YYYY-MM'
    end_month : str
        End month in format 'YYYY-MM'
    
    Returns:
    --------
    pd.DataFrame
        Filtered price DataFrame
    """
    return prices.loc[start_month:end_month]
