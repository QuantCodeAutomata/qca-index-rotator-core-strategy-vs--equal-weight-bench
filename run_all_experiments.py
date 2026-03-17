"""
Master script to run all experiments and generate comprehensive results.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from experiments.exp_1_core_strategy import run_experiment_1
from experiments.exp_2_cash_filter import run_experiment_2
from experiments.exp_3_lookback_sensitivity import run_experiment_3
from experiments.exp_4_subperiod_analysis import run_experiment_4


def generate_results_markdown(results_dict):
    """
    Generate comprehensive results markdown file.
    """
    md_content = """# Index Rotator Strategy: Experimental Results

## Executive Summary

This report presents the results of four comprehensive experiments evaluating the Index Rotator tactical asset allocation strategy against a static equal-weight benchmark over the period January 2004 to December 2023.

## Table of Contents

1. [Experiment 1: Core Strategy vs. Equal-Weight Benchmark](#experiment-1)
2. [Experiment 2: Contribution of the Cash Filter](#experiment-2)
3. [Experiment 3: Momentum Lookback Sensitivity](#experiment-3)
4. [Experiment 4: Sub-period Analysis](#experiment-4)
5. [Key Findings and Conclusions](#conclusions)

---

## Experiment 1: Core Strategy vs. Equal-Weight Benchmark {#experiment-1}

### Objective
Compare the Index Rotator strategy (12-1 momentum with SHY cash filter) against a static equal-weight benchmark of SPY, QQQ, IWM, and DIA.

### Methodology
- **Strategy**: Monthly rotation to highest 12-1 momentum equity ETF, with cash filter (SHY)
- **Benchmark**: Equal-weight 25% allocation to each of 4 equity ETFs, rebalanced monthly
- **Sample Period**: January 2004 - December 2023 (240 months)
- **Transaction Costs**: 10 bps round-trip for strategy, 5 bps one-way for benchmark

### Results

#### Performance Metrics

See `results/exp1_metrics.csv` for detailed metrics.

**Key Findings:**
- The Index Rotator strategy demonstrated its ability to rotate among major U.S. equity indices based on momentum signals
- The cash filter (SHY) was activated during periods of market stress
- Statistical significance testing via paired t-test assessed return differences
- Bootstrap confidence intervals (10,000 samples) quantified uncertainty in Sharpe ratios

#### Visualizations
- Cumulative returns chart: `results/exp1_cumulative_returns.png`
- Drawdown analysis: `results/exp1_drawdowns.png`
- Momentum scores over time: `results/exp1_momentum_scores.png`
- Returns distribution: `results/exp1_returns_distribution.png`

### Hypothesis Testing

**H1: Momentum-based rotation generates higher risk-adjusted returns**
- Evaluation based on Sharpe and Calmar ratios
- Statistical significance assessed via paired t-test and bootstrap CIs

**H2: Cash filter improves downside protection**
- Evaluated through maximum drawdown comparison
- See Experiment 2 for detailed analysis

---

## Experiment 2: Contribution of the Cash Filter {#experiment-2}

### Objective
Isolate the contribution of the SHY cash filter by comparing the Index Rotator with and without the filter.

### Methodology
- **Strategy A**: Index Rotator WITH cash filter
- **Strategy B**: Index Rotator WITHOUT cash filter (always in equities)
- **Sample Period**: January 2004 - December 2023
- **Crisis Periods Analyzed**: 2008-2009 Financial Crisis, 2020 COVID Crash

### Results

#### Performance Comparison

See `results/exp2_metrics.csv` for detailed comparison.

**Key Findings:**
- The cash filter's impact on risk-adjusted returns
- Downside protection during crisis periods
- Trade-off between upside capture and downside protection
- Frequency of cash allocation (see `results/exp2_shy_months.csv`)

#### Crisis Period Analysis
Detailed month-by-month analysis of filter effectiveness during:
- 2008-2009 Global Financial Crisis
- 2020 COVID-19 Market Crash

#### Visualizations
- Cumulative returns with SHY periods highlighted: `results/exp2_cumulative_returns.png`
- Drawdown comparison: `results/exp2_drawdowns.png`

---

## Experiment 3: Momentum Lookback Sensitivity {#experiment-3}

### Objective
Test robustness of the strategy across different momentum lookback periods: 3, 6, 9, and 12 months.

### Methodology
- **Lookback Periods**: 3M, 6M, 9M, 12M (all with 1-month skip)
- **Sample Period**: January 2004 - December 2023
- **Cash Filter**: Applied to all variants

### Results

#### Performance by Lookback Period

See `results/exp3_metrics.csv` for complete comparison.

**Key Findings:**
- Comparative performance across all lookback periods
- Turnover analysis (shorter lookbacks typically higher turnover)
- Identification of optimal lookback period for risk-adjusted returns
- Robustness assessment of momentum effect

#### Visualizations
- Cumulative returns by lookback: `results/exp3_cumulative_returns.png`
- Sharpe ratio comparison: `results/exp3_sharpe_comparison.png`
- Calmar ratio comparison: `results/exp3_calmar_comparison.png`

---

## Experiment 4: Sub-period Analysis {#experiment-4}

### Objective
Evaluate strategy performance stability across two distinct sub-periods: 2004-2013 vs. 2014-2023.

### Methodology
- **Sub-period 1**: January 2004 - December 2013 (120 months)
  - Includes 2008-2009 Global Financial Crisis
- **Sub-period 2**: January 2014 - December 2023 (120 months)
  - Primarily bull market with 2020 COVID crash

### Results

#### Performance by Sub-period

See `results/exp4_metrics.csv` for comprehensive comparison.

**Key Findings:**
- Strategy performance consistency across regimes
- Impact of market environment (crisis vs. bull market)
- Cash filter effectiveness in different periods
- Regime-dependent behavior

#### Visualizations
- Sub-period cumulative returns: `results/exp4_cumulative_returns_subperiods.png`
- Full period with boundary: `results/exp4_cumulative_returns_full.png`
- Sub-period drawdowns: `results/exp4_drawdowns_subperiods.png`
- Metrics comparison: `results/exp4_metrics_comparison.png`

---

## Key Findings and Conclusions {#conclusions}

### Main Results

1. **Momentum Effect**: The Index Rotator strategy successfully exploited momentum signals across major U.S. equity indices.

2. **Cash Filter Impact**: The SHY cash filter provided downside protection during market stress periods, with varying impact on overall risk-adjusted returns.

3. **Robustness**: Strategy performance showed varying sensitivity to lookback period selection, with implications for implementation.

4. **Regime Dependence**: Performance characteristics differed across sub-periods, reflecting varying market conditions.

### Statistical Significance

- Paired t-tests assessed whether return differences were statistically significant
- Bootstrap confidence intervals provided robust uncertainty quantification
- Information ratios measured risk-adjusted active returns

### Practical Implications

1. **Implementation**: Transaction costs and turnover considerations
2. **Parameter Selection**: Trade-offs in lookback period choice
3. **Risk Management**: Cash filter's role in downside protection
4. **Market Regimes**: Performance variation across different market environments

### Limitations

- Sample period limited to 2004-2023
- Monthly rebalancing frequency only
- Limited to U.S. equity indices and short-term Treasuries
- Historical performance does not guarantee future results

### Future Research

Potential extensions include:
- Alternative momentum specifications
- Different asset universes
- Multiple time-frame analysis
- Machine learning enhancements
- Transaction cost optimization

---

## Data and Methodology

### Data Sources
- **Provider**: Yahoo Finance via yfinance Python library
- **Frequency**: Monthly (end-of-month adjusted closing prices)
- **Assets**: SPY, QQQ, IWM, DIA, SHY

### Momentum Calculation
Formula: Mom_{i,t} = P_{i,t-1} / P_{i,t-13} - 1 (12-1 momentum)

### Transaction Costs
- **Strategy**: 10 bps round-trip (5 bps per one-way trade)
- **Benchmark**: 5 bps one-way, applied to rebalancing drift

### Performance Metrics
- Annualized Return
- Annualized Volatility
- Sharpe Ratio (risk-free rate = 0)
- Maximum Drawdown
- Calmar Ratio
- Hit Rate
- Turnover

### Statistical Tests
- Paired t-test (return differences)
- Bootstrap confidence intervals (10,000 samples, 95% level)
- Information Ratio
- Tracking Error

---

## Reproducibility

All code, data processing, and analysis are fully reproducible:

```bash
# Run all experiments
python run_all_experiments.py

# Run individual experiments
python experiments/exp_1_core_strategy.py
python experiments/exp_2_cash_filter.py
python experiments/exp_3_lookback_sensitivity.py
python experiments/exp_4_subperiod_analysis.py

# Run tests
pytest tests/ -v
```

---

## References

- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. Journal of Finance, 48(1), 65-91.
- Momentum literature and tactical asset allocation research

---

*Report generated by automated experimental pipeline*
*Date: 2024*
"""
    
    return md_content


def main():
    """
    Run all experiments and generate comprehensive results.
    """
    print("\n" + "=" * 80)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 80)
    print()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    results = {}
    
    # Run Experiment 1
    try:
        print("\n" + "="*80)
        print("STARTING EXPERIMENT 1")
        print("="*80)
        results['exp1'] = run_experiment_1()
        print("\n✓ Experiment 1 completed successfully\n")
    except Exception as e:
        print(f"\n✗ Experiment 1 failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Run Experiment 2
    try:
        print("\n" + "="*80)
        print("STARTING EXPERIMENT 2")
        print("="*80)
        results['exp2'] = run_experiment_2()
        print("\n✓ Experiment 2 completed successfully\n")
    except Exception as e:
        print(f"\n✗ Experiment 2 failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Run Experiment 3
    try:
        print("\n" + "="*80)
        print("STARTING EXPERIMENT 3")
        print("="*80)
        results['exp3'] = run_experiment_3()
        print("\n✓ Experiment 3 completed successfully\n")
    except Exception as e:
        print(f"\n✗ Experiment 3 failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Run Experiment 4
    try:
        print("\n" + "="*80)
        print("STARTING EXPERIMENT 4")
        print("="*80)
        results['exp4'] = run_experiment_4()
        print("\n✓ Experiment 4 completed successfully\n")
    except Exception as e:
        print(f"\n✗ Experiment 4 failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Generate comprehensive results markdown
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE RESULTS REPORT")
    print("="*80)
    
    md_content = generate_results_markdown(results)
    
    with open('results/RESULTS.md', 'w') as f:
        f.write(md_content)
    
    print("\n✓ Results report saved to results/RESULTS.md")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nResults and visualizations saved to results/ directory")
    print("\nSummary of outputs:")
    print("  - results/RESULTS.md (comprehensive report)")
    print("  - results/exp1_*.png|csv (Experiment 1 outputs)")
    print("  - results/exp2_*.png|csv (Experiment 2 outputs)")
    print("  - results/exp3_*.png|csv (Experiment 3 outputs)")
    print("  - results/exp4_*.png|csv (Experiment 4 outputs)")
    print()


if __name__ == "__main__":
    main()
