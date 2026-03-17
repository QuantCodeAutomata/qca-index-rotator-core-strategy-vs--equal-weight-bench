"""
Experiment 2: Contribution of the Cash Filter

Isolates the contribution of the SHY cash filter by comparing filtered vs. unfiltered 
versions of the Index Rotator strategy.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_strategy_data
from src.strategies import index_rotator_strategy, get_months_in_cash
from src.performance import calculate_all_metrics
from src.visualization import plot_cumulative_returns, plot_drawdowns
from matplotlib.patches import Rectangle


def run_experiment_2():
    """
    Run Experiment 2: Cash Filter Contribution Analysis
    """
    print("=" * 80)
    print("EXPERIMENT 2: CONTRIBUTION OF THE CASH FILTER")
    print("=" * 80)
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    prices = load_strategy_data(start_date='2003-01-01', end_date='2023-12-31')
    print(f"Loaded {len(prices)} months of data")
    print()
    
    # Step 2: Run Index Rotator WITH cash filter
    print("Step 2: Running Index Rotator WITH cash filter...")
    filtered_returns, filtered_assets, _ = index_rotator_strategy(
        prices,
        equity_tickers=['SPY', 'QQQ', 'IWM', 'DIA'],
        cash_ticker='SHY',
        apply_filter=True,
        lookback=12,
        transaction_cost=0.001
    )
    
    filtered_returns = filtered_returns['2004-01':'2023-12']
    filtered_assets = filtered_assets['2004-01':'2023-12']
    
    print(f"Generated {len(filtered_returns)} months of returns")
    print()
    
    # Step 3: Run Index Rotator WITHOUT cash filter
    print("Step 3: Running Index Rotator WITHOUT cash filter...")
    unfiltered_returns, unfiltered_assets, _ = index_rotator_strategy(
        prices,
        equity_tickers=['SPY', 'QQQ', 'IWM', 'DIA'],
        cash_ticker='SHY',
        apply_filter=False,
        lookback=12,
        transaction_cost=0.001
    )
    
    unfiltered_returns = unfiltered_returns['2004-01':'2023-12']
    unfiltered_assets = unfiltered_assets['2004-01':'2023-12']
    
    print(f"Generated {len(unfiltered_returns)} months of returns")
    print()
    
    # Step 4: Calculate performance metrics
    print("Step 4: Calculating performance metrics...")
    
    filtered_metrics = calculate_all_metrics(
        filtered_returns,
        unfiltered_returns,
        periods_per_year=12,
        risk_free_rate=0.0
    )
    
    unfiltered_metrics = calculate_all_metrics(
        unfiltered_returns,
        periods_per_year=12,
        risk_free_rate=0.0
    )
    
    # Count months in SHY
    shy_count, shy_dates = get_months_in_cash(filtered_assets, 'SHY')
    shy_pct = (shy_count / len(filtered_assets)) * 100
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print()
    
    metrics_df = pd.DataFrame({
        'With Cash Filter': filtered_metrics,
        'Without Cash Filter': unfiltered_metrics,
        'Difference': pd.Series(filtered_metrics) - pd.Series(unfiltered_metrics)
    })
    
    print(metrics_df.to_string())
    print()
    print(f"Months in SHY (cash): {shy_count} ({shy_pct:.1f}%)")
    print()
    
    # Step 5: Crisis period analysis
    print("Step 5: Analyzing crisis periods...")
    
    crisis_periods = [
        ('2008 Financial Crisis', '2008-01', '2009-06'),
        ('2020 COVID Crash', '2020-02', '2020-05')
    ]
    
    print("\n" + "=" * 80)
    print("CRISIS PERIOD ANALYSIS")
    print("=" * 80)
    print()
    
    for crisis_name, start, end in crisis_periods:
        print(f"{crisis_name} ({start} to {end}):")
        
        # Calculate cumulative returns
        filtered_crisis = filtered_returns.loc[start:end]
        unfiltered_crisis = unfiltered_returns.loc[start:end]
        
        filtered_cum = (1 + filtered_crisis).prod() - 1
        unfiltered_cum = (1 + unfiltered_crisis).prod() - 1
        
        print(f"  With Cash Filter: {filtered_cum*100:.2f}%")
        print(f"  Without Cash Filter: {unfiltered_cum*100:.2f}%")
        print(f"  Difference: {(filtered_cum - unfiltered_cum)*100:.2f}%")
        
        # Count SHY months during crisis
        crisis_assets = filtered_assets.loc[start:end]
        crisis_shy_count = (crisis_assets == 'SHY').sum()
        print(f"  Months in SHY: {crisis_shy_count}/{len(crisis_assets)}")
        print()
    
    # Step 6: Create visualizations
    print("Step 6: Creating visualizations...")
    
    os.makedirs('results', exist_ok=True)
    
    # Cumulative returns with SHY periods highlighted
    returns_dict = {
        'With Cash Filter': filtered_returns,
        'Without Cash Filter': unfiltered_returns
    }
    
    fig = plot_cumulative_returns(
        returns_dict,
        title="Index Rotator: With vs. Without Cash Filter (2004-2023)",
        save_path=None
    )
    
    # Highlight SHY periods
    ax = fig.axes[0]
    in_shy = filtered_assets == 'SHY'
    
    # Find contiguous SHY periods
    shy_starts = []
    shy_ends = []
    in_period = False
    
    for i, (date, is_shy) in enumerate(in_shy.items()):
        if is_shy and not in_period:
            shy_starts.append(date)
            in_period = True
        elif not is_shy and in_period:
            shy_ends.append(in_shy.index[i-1])
            in_period = False
    
    if in_period:
        shy_ends.append(in_shy.index[-1])
    
    # Add shaded regions
    for start, end in zip(shy_starts, shy_ends):
        ax.axvspan(start, end, alpha=0.2, color='gray')
    
    # Add legend entry for SHY periods
    ax.add_patch(Rectangle((0, 0), 0, 0, fc='gray', alpha=0.2, label='In SHY (Cash)'))
    ax.legend(loc='best')
    
    plt.savefig('results/exp2_cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Drawdowns
    plot_drawdowns(
        returns_dict,
        title="Drawdowns: With vs. Without Cash Filter (2004-2023)",
        save_path='results/exp2_drawdowns.png'
    )
    plt.close()
    
    print("Visualizations saved to results/ directory")
    print()
    
    # Step 7: Save results
    print("Step 7: Saving results...")
    
    metrics_df.to_csv('results/exp2_metrics.csv')
    
    # Save SHY months
    shy_df = pd.DataFrame({
        'Date': shy_dates,
        'In_SHY': True
    })
    shy_df.to_csv('results/exp2_shy_months.csv', index=False)
    
    print("Results saved to results/exp2_metrics.csv and results/exp2_shy_months.csv")
    print()
    
    print("=" * 80)
    print("EXPERIMENT 2 COMPLETED")
    print("=" * 80)
    
    return {
        'metrics': metrics_df,
        'shy_count': shy_count,
        'shy_percentage': shy_pct
    }


if __name__ == "__main__":
    results = run_experiment_2()
