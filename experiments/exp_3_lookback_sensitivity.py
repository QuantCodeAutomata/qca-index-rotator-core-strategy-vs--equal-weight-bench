"""
Experiment 3: Robustness Check - Momentum Lookback Period Sensitivity

Tests sensitivity of the Index Rotator strategy to different momentum lookback periods 
(3, 6, 9, 12 months).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_strategy_data
from src.strategies import index_rotator_strategy, equal_weight_benchmark, calculate_turnover
from src.performance import calculate_all_metrics
from src.visualization import plot_cumulative_returns


def run_experiment_3():
    """
    Run Experiment 3: Momentum Lookback Sensitivity Analysis
    """
    print("=" * 80)
    print("EXPERIMENT 3: MOMENTUM LOOKBACK PERIOD SENSITIVITY")
    print("=" * 80)
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    prices = load_strategy_data(start_date='2003-01-01', end_date='2023-12-31')
    print(f"Loaded {len(prices)} months of data")
    print()
    
    # Step 2: Define lookback periods
    lookback_periods = [3, 6, 9, 12]
    
    # Dictionary to store results
    all_returns = {}
    all_assets = {}
    all_metrics = {}
    
    # Step 3: Run strategy for each lookback period
    print("Step 3: Running strategy for each lookback period...")
    print()
    
    for lookback in lookback_periods:
        print(f"  Running with {lookback}-month lookback...")
        
        returns, assets, _ = index_rotator_strategy(
            prices,
            equity_tickers=['SPY', 'QQQ', 'IWM', 'DIA'],
            cash_ticker='SHY',
            apply_filter=True,
            lookback=lookback,
            transaction_cost=0.001
        )
        
        # Filter to Jan 2004 onwards
        returns = returns['2004-01':'2023-12']
        assets = assets['2004-01':'2023-12']
        
        all_returns[f'{lookback}M Lookback'] = returns
        all_assets[f'{lookback}M Lookback'] = assets
        
        print(f"    Generated {len(returns)} months of returns")
    
    print()
    
    # Step 4: Run benchmark
    print("Step 4: Running Equal-Weight Benchmark...")
    benchmark_returns, _ = equal_weight_benchmark(
        prices,
        tickers=['SPY', 'QQQ', 'IWM', 'DIA'],
        transaction_cost_bps=0.0005
    )
    benchmark_returns = benchmark_returns['2004-01':'2023-12']
    all_returns['Equal-Weight Benchmark'] = benchmark_returns
    print()
    
    # Step 5: Calculate performance metrics
    print("Step 5: Calculating performance metrics...")
    
    for name, returns in all_returns.items():
        metrics = calculate_all_metrics(
            returns,
            benchmark_returns if name != 'Equal-Weight Benchmark' else None,
            periods_per_year=12,
            risk_free_rate=0.0
        )
        
        # Add turnover for strategies (not benchmark)
        if name != 'Equal-Weight Benchmark':
            turnover = calculate_turnover(all_assets[name])
            metrics['Turnover (%)'] = turnover
            
            # Count SHY months
            shy_count = (all_assets[name] == 'SHY').sum()
            shy_pct = (shy_count / len(all_assets[name])) * 100
            metrics['Months in SHY (%)'] = shy_pct
        
        all_metrics[name] = metrics
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS BY LOOKBACK PERIOD")
    print("=" * 80)
    print()
    
    metrics_df = pd.DataFrame(all_metrics).T
    
    # Reorder columns
    col_order = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 
                 'Maximum Drawdown', 'Calmar Ratio']
    if 'Hit Rate (%)' in metrics_df.columns:
        col_order.append('Hit Rate (%)')
    if 'Turnover (%)' in metrics_df.columns:
        col_order.append('Turnover (%)')
    if 'Months in SHY (%)' in metrics_df.columns:
        col_order.append('Months in SHY (%)')
    
    metrics_df = metrics_df[col_order]
    
    print(metrics_df.to_string())
    print()
    
    # Step 6: Identify best performing lookback
    print("Step 6: Identifying best performing lookback...")
    
    # Exclude benchmark from comparison
    strategy_metrics = metrics_df.drop('Equal-Weight Benchmark')
    
    best_sharpe = strategy_metrics['Sharpe Ratio'].idxmax()
    best_calmar = strategy_metrics['Calmar Ratio'].idxmax()
    best_return = strategy_metrics['Annualized Return'].idxmax()
    lowest_mdd = strategy_metrics['Maximum Drawdown'].idxmin()
    
    print(f"Best Sharpe Ratio: {best_sharpe}")
    print(f"Best Calmar Ratio: {best_calmar}")
    print(f"Highest Return: {best_return}")
    print(f"Lowest Max Drawdown: {lowest_mdd}")
    print()
    
    # Step 7: Create visualizations
    print("Step 7: Creating visualizations...")
    
    os.makedirs('results', exist_ok=True)
    
    # Cumulative returns for all variants
    plot_cumulative_returns(
        all_returns,
        title="Cumulative Returns by Momentum Lookback Period (2004-2023)",
        save_path='results/exp3_cumulative_returns.png'
    )
    plt.close()
    
    # Bar chart comparing Sharpe ratios
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sharpe_values = metrics_df['Sharpe Ratio']
    colors = ['#1f77b4' if 'Lookback' in name else '#ff7f0e' for name in sharpe_values.index]
    
    ax.bar(range(len(sharpe_values)), sharpe_values.values, color=colors)
    ax.set_xticks(range(len(sharpe_values)))
    ax.set_xticklabels(sharpe_values.index, rotation=45, ha='right')
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Sharpe Ratio Comparison by Lookback Period', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/exp3_sharpe_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bar chart comparing Calmar ratios
    fig, ax = plt.subplots(figsize=(12, 6))
    
    calmar_values = metrics_df['Calmar Ratio']
    
    ax.bar(range(len(calmar_values)), calmar_values.values, color=colors)
    ax.set_xticks(range(len(calmar_values)))
    ax.set_xticklabels(calmar_values.index, rotation=45, ha='right')
    ax.set_ylabel('Calmar Ratio', fontsize=12)
    ax.set_title('Calmar Ratio Comparison by Lookback Period', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/exp3_calmar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to results/ directory")
    print()
    
    # Step 8: Save results
    print("Step 8: Saving results...")
    
    metrics_df.to_csv('results/exp3_metrics.csv')
    
    print("Results saved to results/exp3_metrics.csv")
    print()
    
    print("=" * 80)
    print("EXPERIMENT 3 COMPLETED")
    print("=" * 80)
    
    return {
        'metrics': metrics_df,
        'best_sharpe': best_sharpe,
        'best_calmar': best_calmar
    }


if __name__ == "__main__":
    results = run_experiment_3()
