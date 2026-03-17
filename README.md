# Index Rotator Core Strategy vs. Equal-Weight Benchmark

## Project Overview

This repository implements a comprehensive quantitative analysis of the Index Rotator tactical asset allocation strategy compared to a static equal-weight benchmark. The strategy uses 12-1 momentum to rotate monthly among major U.S. equity indices (SPY, QQQ, IWM, DIA) with a SHY cash filter.

## Research Objectives

1. **Test H1**: A momentum-based rotation strategy across major U.S. equity indices generates higher risk-adjusted returns than a static equal-weight benchmark.
2. **Test H2**: A cash filter (using SHY as a safe-haven alternative) improves the strategy's downside protection without significantly reducing upside capture.

## Experiments

### Experiment 1: Core Strategy vs. Benchmark
Full sample analysis (Jan 2004 - Dec 2023) comparing the Index Rotator strategy against an equal-weight benchmark.

### Experiment 2: Cash Filter Contribution
Isolates the contribution of the SHY cash filter by comparing filtered vs. unfiltered versions.

### Experiment 3: Momentum Lookback Sensitivity
Tests robustness across different momentum lookback periods (3, 6, 9, 12 months).

### Experiment 4: Sub-period Analysis
Evaluates performance stability across two sub-periods (2004-2013 vs. 2014-2023).

## Repository Structure

```
├── src/
│   ├── data_loader.py          # Data download and preprocessing
│   ├── momentum.py              # Momentum signal calculation
│   ├── strategies.py            # Strategy implementation
│   ├── performance.py           # Performance metrics calculation
│   ├── statistical_tests.py    # Statistical testing utilities
│   └── visualization.py         # Plotting functions
├── tests/
│   ├── test_data_loader.py
│   ├── test_momentum.py
│   ├── test_strategies.py
│   ├── test_performance.py
│   └── test_statistical_tests.py
├── experiments/
│   ├── exp_1_core_strategy.py
│   ├── exp_2_cash_filter.py
│   ├── exp_3_lookback_sensitivity.py
│   └── exp_4_subperiod_analysis.py
├── results/
│   └── RESULTS.md
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:
```bash
python experiments/exp_1_core_strategy.py
python experiments/exp_2_cash_filter.py
python experiments/exp_3_lookback_sensitivity.py
python experiments/exp_4_subperiod_analysis.py
```

Run tests:
```bash
pytest tests/ -v
```

## Data Requirements

Monthly adjusted closing prices for five ETFs:
- SPY (SPDR S&P 500 ETF Trust)
- QQQ (Invesco Nasdaq-100 ETF)
- IWM (iShares Russell 2000 ETF)
- DIA (SPDR Dow Jones Industrial Average ETF)
- SHY (iShares 1-3 Year Treasury Bond ETF)

Data sourced from Yahoo Finance via yfinance library.
Sample period: January 2004 – December 2023 (240 months).

## Methodology Highlights

- **Momentum Signal**: 12-1 momentum (11-month return with 1-month skip)
- **Asset Selection**: Monthly rotation to highest momentum equity ETF
- **Cash Filter**: Switch to SHY when selected equity momentum <= SHY momentum
- **Transaction Costs**: 10 bps round-trip (5 bps per one-way trade)
- **Benchmark**: Equal-weight 25% each in SPY, QQQ, IWM, DIA (monthly rebalanced)

## Results

See [results/RESULTS.md](results/RESULTS.md) for detailed findings including:
- Performance metrics (returns, volatility, Sharpe, MDD, Calmar)
- Statistical significance tests
- Bootstrap confidence intervals
- Visualization charts

## License

MIT License
