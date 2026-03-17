"""
Visualization module.
Contains functions for creating charts and plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_cumulative_returns(
    returns_dict: Dict[str, pd.Series],
    title: str = "Cumulative Returns",
    save_path: str = None
) -> plt.Figure:
    """
    Plot cumulative returns for multiple strategies.
    
    Parameters:
    -----------
    returns_dict : Dict[str, pd.Series]
        Dictionary mapping strategy names to return series
    title : str
        Plot title
    save_path : str
        Path to save figure (optional)
    
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Wealth ($1 initial)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_drawdowns(
    returns_dict: Dict[str, pd.Series],
    title: str = "Drawdowns",
    save_path: str = None
) -> plt.Figure:
    """
    Plot drawdown series for multiple strategies.
    
    Parameters:
    -----------
    returns_dict : Dict[str, pd.Series]
        Dictionary mapping strategy names to return series
    title : str
        Plot title
    save_path : str
        Path to save figure (optional)
    
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        ax.plot(drawdown.index, drawdown.values * 100, label=name, linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_momentum_scores(
    momentum: pd.DataFrame,
    tickers: List[str],
    title: str = "Momentum Scores Over Time",
    save_path: str = None
) -> plt.Figure:
    """
    Plot momentum scores over time for multiple assets.
    
    Parameters:
    -----------
    momentum : pd.DataFrame
        DataFrame with momentum scores
    tickers : List[str]
        List of tickers to plot
    title : str
        Plot title
    save_path : str
        Path to save figure (optional)
    
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for ticker in tickers:
        if ticker in momentum.columns:
            ax.plot(momentum.index, momentum[ticker] * 100, label=ticker, linewidth=1.5, alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Momentum Score (%)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_asset_allocation(
    selected_assets: pd.Series,
    title: str = "Asset Allocation Over Time",
    save_path: str = None
) -> plt.Figure:
    """
    Plot asset allocation over time as a stacked area chart.
    
    Parameters:
    -----------
    selected_assets : pd.Series
        Series of selected assets over time
    title : str
        Plot title
    save_path : str
        Path to save figure (optional)
    
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Count allocations
    allocation_counts = selected_assets.value_counts()
    
    # Create dummy variable matrix
    unique_assets = selected_assets.unique()
    allocation_matrix = pd.DataFrame(index=selected_assets.index, columns=unique_assets)
    
    for date, asset in selected_assets.items():
        allocation_matrix.loc[date] = 0
        allocation_matrix.loc[date, asset] = 1
    
    # Plot stacked area
    allocation_matrix.plot.area(ax=ax, alpha=0.7, linewidth=0)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Allocation', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_returns_distribution(
    returns_dict: Dict[str, pd.Series],
    title: str = "Returns Distribution",
    save_path: str = None
) -> plt.Figure:
    """
    Plot return distributions as histograms.
    
    Parameters:
    -----------
    returns_dict : Dict[str, pd.Series]
        Dictionary mapping strategy names to return series
    title : str
        Plot title
    save_path : str
        Path to save figure (optional)
    
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, returns in returns_dict.items():
        ax.hist(returns.dropna() * 100, bins=30, alpha=0.5, label=name, edgecolor='black')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Monthly Return (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rolling_sharpe(
    returns_dict: Dict[str, pd.Series],
    window: int = 36,
    title: str = "Rolling 3-Year Sharpe Ratio",
    save_path: str = None
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio.
    
    Parameters:
    -----------
    returns_dict : Dict[str, pd.Series]
        Dictionary mapping strategy names to return series
    window : int
        Rolling window size in months (default 36 = 3 years)
    title : str
        Plot title
    save_path : str
        Path to save figure (optional)
    
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, returns in returns_dict.items():
        rolling_mean = returns.rolling(window).mean() * 12
        rolling_std = returns.rolling(window).std() * np.sqrt(12)
        rolling_sharpe = rolling_mean / rolling_std
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_crisis_performance(
    returns_dict: Dict[str, pd.Series],
    crisis_periods: List[Tuple[str, str, str]],
    save_path: str = None
) -> plt.Figure:
    """
    Plot cumulative returns during specific crisis periods.
    
    Parameters:
    -----------
    returns_dict : Dict[str, pd.Series]
        Dictionary mapping strategy names to return series
    crisis_periods : List[Tuple[str, str, str]]
        List of (name, start_date, end_date) tuples
    save_path : str
        Path to save figure (optional)
    
    Returns:
    --------
    plt.Figure
        Figure object
    """
    n_periods = len(crisis_periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(7 * n_periods, 6))
    
    if n_periods == 1:
        axes = [axes]
    
    for i, (crisis_name, start, end) in enumerate(crisis_periods):
        ax = axes[i]
        
        for name, returns in returns_dict.items():
            crisis_returns = returns.loc[start:end]
            cumulative = (1 + crisis_returns).cumprod()
            ax.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
        
        ax.set_title(f'{crisis_name}\n({start} to {end})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Wealth', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_bootstrap_distribution(
    bootstrap_samples: Dict[str, np.ndarray],
    title: str = "Bootstrap Sharpe Ratio Distribution",
    save_path: str = None
) -> plt.Figure:
    """
    Plot bootstrap distribution of Sharpe ratios.
    
    Parameters:
    -----------
    bootstrap_samples : Dict[str, np.ndarray]
        Dictionary mapping strategy names to bootstrap sample arrays
    title : str
        Plot title
    save_path : str
        Path to save figure (optional)
    
    Returns:
    --------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, samples in bootstrap_samples.items():
        ax.hist(samples, bins=50, alpha=0.5, label=name, edgecolor='black')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
