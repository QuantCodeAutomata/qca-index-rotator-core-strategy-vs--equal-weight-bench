"""
Experiment 1: Index Rotator Core Strategy vs. Equal-Weight Benchmark

Full sample analysis (Jan 2004 - Dec 2023) comparing the Index Rotator strategy 
against an equal-weight benchmark.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_strategy_data
from src.strategies import index_rotator_strategy, equal_weight_benchmark, calculate_turnover
from src.performance import calculate_all_metrics, calculate_cumulative_returns
from src.statistical_tests import statistical_summary
from src.visualization import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_momentum_scores,
    plot_returns_distribution
)


def run_experiment_1():
    """
    Run Experiment 1: Core Strategy vs. Equal-Weight Benchmark
    """
    print("=" * 80)
    print("EXPERIMENT 1: INDEX ROTATOR CORE STRATEGY VS. EQUAL-WEIGHT BENCHMARK")
    print("=" * 80)
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    prices = load_strategy_data(start_date='2003-01-01', end_date='2023-12-31')
    print(f"Loaded {len(prices)} months of data from {prices.index[0]} to {prices.index[-1]}")
    print()
    
    # Step 2: Run Index Rotator strategy (with cash filter, 12-1 momentum)
    print("Step 2: Running Index Rotator strategy...")
    rotator_returns, rotator_assets, momentum = index_rotator_strategy(
        prices,
        equity_tickers=['SPY', 'QQQ', 'IWM', 'DIA'],
        cash_ticker='SHY',
        apply_filter=True,
        lookback=12,
        transaction_cost=0.001
    )
    
    # Filter to start from Jan 2004
    rotator_returns = rotator_returns['2004-01':'2023-12']
    rotator_assets = rotator_assets['2004-01':'2023-12']
    
    print(f"Strategy generated {len(rotator_returns)} months of returns")
    print()
    
    # Step 3: Run Equal-Weight Benchmark
    print("Step 3: Running Equal-Weight Benchmark...")
    benchmark_returns, benchmark_weights = equal_weight_benchmark(
        prices,
        tickers=['SPY', 'QQQ', 'IWM', 'DIA'],
        transaction_cost_bps=0.0005
    )
    
    # Filter to start from Jan 2004
    benchmark_returns = benchmark_returns['2004-01':'2023-12']
    
    print(f"Benchmark generated {len(benchmark_returns)} months of returns")
    print()
    
    # Step 4: Calculate performance metrics
    print("Step 4: Calculating performance metrics...")
    
    rotator_metrics = calculate_all_metrics(
        rotator_returns,
        benchmark_returns,
        periods_per_year=12,
        risk_free_rate=0.0
    )
    
    benchmark_metrics = calculate_all_metrics(
        benchmark_returns,
        periods_per_year=12,
        risk_free_rate=0.0
    )
    
    # Calculate turnover
    rotator_turnover = calculate_turnover(rotator_assets)
    rotator_metrics['Turnover (%)'] = rotator_turnover
    
    # Count months in SHY
    shy_months = (rotator_assets == 'SHY').sum()
    shy_pct = (shy_months / len(rotator_assets)) * 100
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print()
    
    # Create comparison table
    metrics_df = pd.DataFrame({
        'Index Rotator': rotator_metrics,
        'Equal-Weight Benchmark': benchmark_metrics
    })
    
    print(metrics_df.to_string())
    print()
    print(f"Months in SHY (cash): {shy_months} ({shy_pct:.1f}%)")
    print()
    
    # Step 5: Statistical tests
    print("Step 5: Performing statistical tests...")
    
    stats_summary = statistical_summary(
        rotator_returns,
        benchmark_returns,
        n_bootstrap=10000,
        confidence_level=0.95,
        random_seed=42
    )
    
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    print()
    
    print("Paired t-test (Return Difference):")
    print(f"  t-statistic: {stats_summary['Paired t-test']['t-statistic']:.4f}")
    print(f"  p-value: {stats_summary['Paired t-test']['p-value']:.4f}")
    print(f"  Significant (p<0.05): {stats_summary['Paired t-test']['significant (p<0.05)']}")
    print()
    
    print("Bootstrap 95% Confidence Intervals for Sharpe Ratio:")
    print(f"  Index Rotator: [{stats_summary['Bootstrap Sharpe CI (Strategy)']['lower_bound']:.4f}, "
          f"{stats_summary['Bootstrap Sharpe CI (Strategy)']['upper_bound']:.4f}]")
    print(f"  Benchmark: [{stats_summary['Bootstrap Sharpe CI (Benchmark)']['lower_bound']:.4f}, "
          f"{stats_summary['Bootstrap Sharpe CI (Benchmark)']['upper_bound']:.4f}]")
    print()
    
    print(f"Information Ratio: {stats_summary['Information Ratio']:.4f}")
    print(f"Tracking Error: {stats_summary['Tracking Error']:.4f}")
    print()
    
    # Step 6: Create visualizations
    print("Step 6: Creating visualizations...")
    
    os.makedirs('results', exist_ok=True)
    
    # Cumulative returns
    returns_dict = {
        'Index Rotator': rotator_returns,
        'Equal-Weight Benchmark': benchmark_returns
    }
    
    plot_cumulative_returns(
        returns_dict,
        title="Cumulative Returns: Index Rotator vs. Equal-Weight Benchmark (2004-2023)",
        save_path='results/exp1_cumulative_returns.png'
    )
    plt.close()
    
    # Drawdowns
    plot_drawdowns(
        returns_dict,
        title="Drawdowns: Index Rotator vs. Equal-Weight Benchmark (2004-2023)",
        save_path='results/exp1_drawdowns.png'
    )
    plt.close()
    
    # Momentum scores
    momentum_filtered = momentum['2004-01':'2023-12']
    plot_momentum_scores(
        momentum_filtered,
        tickers=['SPY', 'QQQ', 'IWM', 'DIA', 'SHY'],
        title="12-1 Momentum Scores Over Time (2004-2023)",
        save_path='results/exp1_momentum_scores.png'
    )
    plt.close()
    
    # Returns distribution
    plot_returns_distribution(
        returns_dict,
        title="Monthly Returns Distribution (2004-2023)",
        save_path='results/exp1_returns_distribution.png'
    )
    plt.close()
    
    print("Visualizations saved to results/ directory")
    print()
    
    # Step 7: Save results
    print("Step 7: Saving results...")
    
    results = {
        'metrics': metrics_df,
        'statistical_tests': stats_summary,
        'shy_allocation': {
            'months': shy_months,
            'percentage': shy_pct
        }
    }
    
    # Save to CSV
    metrics_df.to_csv('results/exp1_metrics.csv')
    
    print("Results saved to results/exp1_metrics.csv")
    print()
    
    print("=" * 80)
    print("EXPERIMENT 1 COMPLETED")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_experiment_1()
