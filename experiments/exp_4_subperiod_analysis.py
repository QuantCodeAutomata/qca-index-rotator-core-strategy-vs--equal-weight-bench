"""
Experiment 4: Robustness Check - Sub-period Analysis

Splits the full sample period into two equal sub-periods (2004-2013 and 2014-2023) 
and evaluates the Index Rotator strategy independently in each sub-period.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_strategy_data
from src.strategies import index_rotator_strategy, equal_weight_benchmark
from src.performance import calculate_all_metrics
from src.visualization import plot_cumulative_returns, plot_drawdowns


def run_experiment_4():
    """
    Run Experiment 4: Sub-period Analysis (2004-2013 vs. 2014-2023)
    """
    print("=" * 80)
    print("EXPERIMENT 4: SUB-PERIOD ANALYSIS")
    print("=" * 80)
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    prices = load_strategy_data(start_date='2003-01-01', end_date='2023-12-31')
    print(f"Loaded {len(prices)} months of data")
    print()
    
    # Step 2: Run strategies for full period
    print("Step 2: Running strategies for full period...")
    
    rotator_returns, rotator_assets, _ = index_rotator_strategy(
        prices,
        equity_tickers=['SPY', 'QQQ', 'IWM', 'DIA'],
        cash_ticker='SHY',
        apply_filter=True,
        lookback=12,
        transaction_cost=0.001
    )
    
    benchmark_returns, _ = equal_weight_benchmark(
        prices,
        tickers=['SPY', 'QQQ', 'IWM', 'DIA'],
        transaction_cost_bps=0.0005
    )
    
    # Filter to Jan 2004 onwards
    rotator_returns = rotator_returns['2004-01':'2023-12']
    rotator_assets = rotator_assets['2004-01':'2023-12']
    benchmark_returns = benchmark_returns['2004-01':'2023-12']
    
    print(f"Generated {len(rotator_returns)} months of returns")
    print()
    
    # Step 3: Define sub-periods
    print("Step 3: Defining sub-periods...")
    
    subperiods = [
        ('Sub-period 1 (2004-2013)', '2004-01', '2013-12'),
        ('Sub-period 2 (2014-2023)', '2014-01', '2023-12')
    ]
    
    print(f"  Sub-period 1: 2004-01 to 2013-12 (120 months)")
    print(f"  Sub-period 2: 2014-01 to 2023-12 (120 months)")
    print()
    
    # Step 4: Calculate metrics for each sub-period
    print("Step 4: Calculating performance metrics by sub-period...")
    print()
    
    all_metrics = {}
    
    for period_name, start, end in subperiods:
        print(f"  Analyzing {period_name}...")
        
        # Slice returns
        rotator_sub = rotator_returns.loc[start:end]
        benchmark_sub = benchmark_returns.loc[start:end]
        
        # Calculate metrics
        rotator_metrics = calculate_all_metrics(
            rotator_sub,
            benchmark_sub,
            periods_per_year=12,
            risk_free_rate=0.0
        )
        
        benchmark_metrics = calculate_all_metrics(
            benchmark_sub,
            periods_per_year=12,
            risk_free_rate=0.0
        )
        
        # Count SHY months
        assets_sub = rotator_assets.loc[start:end]
        shy_count = (assets_sub == 'SHY').sum()
        shy_pct = (shy_count / len(assets_sub)) * 100
        rotator_metrics['Months in SHY (%)'] = shy_pct
        
        all_metrics[f'{period_name} - Rotator'] = rotator_metrics
        all_metrics[f'{period_name} - Benchmark'] = benchmark_metrics
    
    print()
    
    # Step 5: Create comparison tables
    print("Step 5: Creating comparison tables...")
    print()
    
    metrics_df = pd.DataFrame(all_metrics).T
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS BY SUB-PERIOD")
    print("=" * 80)
    print()
    print(metrics_df.to_string())
    print()
    
    # Create separate tables for easier comparison
    print("\n" + "=" * 80)
    print("SUB-PERIOD 1 (2004-2013): INDEX ROTATOR VS. BENCHMARK")
    print("=" * 80)
    print()
    
    sp1_comparison = pd.DataFrame({
        'Index Rotator': all_metrics['Sub-period 1 (2004-2013) - Rotator'],
        'Equal-Weight Benchmark': all_metrics['Sub-period 1 (2004-2013) - Benchmark']
    })
    print(sp1_comparison.to_string())
    print()
    
    print("\n" + "=" * 80)
    print("SUB-PERIOD 2 (2014-2023): INDEX ROTATOR VS. BENCHMARK")
    print("=" * 80)
    print()
    
    sp2_comparison = pd.DataFrame({
        'Index Rotator': all_metrics['Sub-period 2 (2014-2023) - Rotator'],
        'Equal-Weight Benchmark': all_metrics['Sub-period 2 (2014-2023) - Benchmark']
    })
    print(sp2_comparison.to_string())
    print()
    
    # Step 6: Cross-period comparison
    print("\n" + "=" * 80)
    print("CROSS-PERIOD COMPARISON: INDEX ROTATOR")
    print("=" * 80)
    print()
    
    rotator_cross = pd.DataFrame({
        'Sub-period 1 (2004-2013)': all_metrics['Sub-period 1 (2004-2013) - Rotator'],
        'Sub-period 2 (2014-2023)': all_metrics['Sub-period 2 (2014-2023) - Rotator']
    })
    print(rotator_cross.to_string())
    print()
    
    # Step 7: Create visualizations
    print("Step 7: Creating visualizations...")
    
    os.makedirs('results', exist_ok=True)
    
    # Cumulative returns for each sub-period
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (period_name, start, end) in enumerate(subperiods):
        ax = axes[i]
        
        rotator_sub = rotator_returns.loc[start:end]
        benchmark_sub = benchmark_returns.loc[start:end]
        
        # Calculate cumulative wealth
        rotator_cum = (1 + rotator_sub).cumprod()
        benchmark_cum = (1 + benchmark_sub).cumprod()
        
        ax.plot(rotator_cum.index, rotator_cum.values, label='Index Rotator', linewidth=2)
        ax.plot(benchmark_cum.index, benchmark_cum.values, label='Equal-Weight Benchmark', linewidth=2)
        
        ax.set_title(period_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Wealth ($1 initial)', fontsize=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/exp4_cumulative_returns_subperiods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Full period with sub-period boundary
    returns_dict = {
        'Index Rotator': rotator_returns,
        'Equal-Weight Benchmark': benchmark_returns
    }
    
    fig = plot_cumulative_returns(
        returns_dict,
        title="Cumulative Returns: Full Period with Sub-period Boundary",
        save_path=None
    )
    
    # Add vertical line at sub-period boundary
    ax = fig.axes[0]
    ax.axvline(x=pd.Timestamp('2014-01-01'), color='red', linestyle='--', linewidth=2, label='Sub-period Boundary')
    ax.legend(loc='best')
    
    plt.savefig('results/exp4_cumulative_returns_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Drawdowns by sub-period
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (period_name, start, end) in enumerate(subperiods):
        ax = axes[i]
        
        rotator_sub = rotator_returns.loc[start:end]
        benchmark_sub = benchmark_returns.loc[start:end]
        
        # Calculate drawdowns
        rotator_cum = (1 + rotator_sub).cumprod()
        rotator_max = rotator_cum.cummax()
        rotator_dd = (rotator_cum - rotator_max) / rotator_max
        
        benchmark_cum = (1 + benchmark_sub).cumprod()
        benchmark_max = benchmark_cum.cummax()
        benchmark_dd = (benchmark_cum - benchmark_max) / benchmark_max
        
        ax.plot(rotator_dd.index, rotator_dd.values * 100, label='Index Rotator', linewidth=2)
        ax.plot(benchmark_dd.index, benchmark_dd.values * 100, label='Equal-Weight Benchmark', linewidth=2)
        
        ax.set_title(f'Drawdowns: {period_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/exp4_drawdowns_subperiods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Comparison bar charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ('Annualized Return', 'Annualized Return'),
        ('Sharpe Ratio', 'Sharpe Ratio'),
        ('Maximum Drawdown', 'Maximum Drawdown'),
        ('Calmar Ratio', 'Calmar Ratio')
    ]
    
    for idx, (metric_name, metric_key) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        sp1_rotator = all_metrics['Sub-period 1 (2004-2013) - Rotator'][metric_key]
        sp1_bench = all_metrics['Sub-period 1 (2004-2013) - Benchmark'][metric_key]
        sp2_rotator = all_metrics['Sub-period 2 (2014-2023) - Rotator'][metric_key]
        sp2_bench = all_metrics['Sub-period 2 (2014-2023) - Benchmark'][metric_key]
        
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, [sp1_rotator, sp2_rotator], width, label='Index Rotator')
        ax.bar(x + width/2, [sp1_bench, sp2_bench], width, label='Equal-Weight Benchmark')
        
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['2004-2013', '2014-2023'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/exp4_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to results/ directory")
    print()
    
    # Step 8: Save results
    print("Step 8: Saving results...")
    
    metrics_df.to_csv('results/exp4_metrics.csv')
    sp1_comparison.to_csv('results/exp4_sp1_comparison.csv')
    sp2_comparison.to_csv('results/exp4_sp2_comparison.csv')
    
    print("Results saved to results/ directory")
    print()
    
    print("=" * 80)
    print("EXPERIMENT 4 COMPLETED")
    print("=" * 80)
    
    return {
        'metrics': metrics_df,
        'sp1_comparison': sp1_comparison,
        'sp2_comparison': sp2_comparison
    }


if __name__ == "__main__":
    results = run_experiment_4()
