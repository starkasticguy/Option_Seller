# BankNifty Options Backtesting Framework

A comprehensive backtesting framework for testing BankNifty options selling strategies with realistic option pricing, transaction costs, and market conditions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Output Files](#output-files)
- [Advanced Features](#advanced-features)

## 🎯 Overview

This backtesting framework simulates a delta-neutral short strangle strategy on BankNifty weekly options using historical data and Black-Scholes option pricing. It includes:

- Historical data collection from Zerodha (or mock data generation)
- Black-Scholes option pricing with volatility smile/skew
- Realistic transaction costs (STT, brokerage, slippage)
- Comprehensive performance analytics
- Monte Carlo simulations for robustness testing

## ✨ Features

### Option Pricing
- **Black-Scholes Model**: European option pricing with all Greeks
- **Volatility Smile/Skew**: Adjusts IV based on moneyness (OTM puts have higher IV)
- **Bid-Ask Spreads**: Simulates realistic spreads based on moneyness
- **Greeks Calculation**: Delta, Gamma, Theta, Vega, Rho

### Strategy Simulation
- **Entry Rules**: VIX range, IV percentile, delta targets
- **Exit Rules**: Profit target (50%), stop loss (200%), delta breach
- **Position Management**: Real-time P&L tracking, Greeks monitoring
- **Time Management**: No trading in first/last 15 minutes, expiry day exits

### Transaction Costs
- **Brokerage**: ₹40 per lot (round trip)
- **STT**: 0.0125% on sell side
- **Exchange Charges**: 0.005%
- **GST**: 18% on brokerage
- **Slippage**: Entry (₹2), Exit (₹3), Stop loss (₹5)

### Analytics
- **Performance Metrics**: Total return, CAGR, Sharpe, Sortino, Calmar ratios
- **Trade Analysis**: Win rate, profit factor, avg trade duration
- **Drawdown Analysis**: Max drawdown, recovery time
- **Monthly Returns**: Heatmap visualization
- **Interactive Reports**: HTML reports with Plotly charts

### Monte Carlo Simulations
- **Parameter Variations**: Test different profit targets, stop losses, delta targets
- **Market Regimes**: Bull, bear, ranging, high volatility scenarios
- **Stress Tests**: Gap moves, volatility spikes
- **Bootstrap Analysis**: Confidence intervals for expected returns

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Data Directories

The framework will automatically create required directories on first run:
- `data/historical/` - Historical market data
- `data/backtest_results/` - Backtest results and reports
- `data/cache/` - Cached data

## 🏃 Quick Start

### Basic Backtest

```bash
# Run backtest with default parameters
python run_backtest.py --collect-data --load-data

# Run with custom date range
python run_backtest.py --start-date 2023-01-01 --end-date 2023-12-31

# Run with custom strategy parameters
python run_backtest.py \
    --strategy SHORT_STRANGLE \
    --initial-capital 500000 \
    --profit-target 0.60 \
    --stop-loss 2.5 \
    --position-size 2
```

### With Monte Carlo Simulations

```bash
python run_backtest.py \
    --load-data \
    --run-monte-carlo
```

### Command Line Options

```
--start-date YYYY-MM-DD       Start date for backtest
--end-date YYYY-MM-DD         End date for backtest
--interval 5minute            Data interval (5minute, 15minute, day)
--collect-data                Collect new historical data
--load-data                   Load existing data
--skip-backtest               Only collect data, skip backtest
--run-monte-carlo             Run Monte Carlo simulations
--strategy SHORT_STRANGLE     Strategy type
--initial-capital 300000      Initial capital
--position-size 1             Number of lots
--profit-target 0.5           Profit target (50%)
--stop-loss 2.0               Stop loss (200%)
--output-dir /path/to/output  Custom output directory
```

## 📚 Module Documentation

### 1. `backtest_config.py`
Configuration management for all backtesting parameters.

**Key Settings:**
- `STRATEGY_CONFIG`: Strategy parameters (entry/exit rules, position sizing)
- `COST_CONFIG`: Transaction costs (brokerage, STT, slippage)
- `OPTION_PRICING_CONFIG`: Black-Scholes parameters, volatility skew
- `MONTE_CARLO_CONFIG`: Monte Carlo simulation settings

### 2. `data_collector.py`
Historical data collection and processing.

**Features:**
- Download BankNifty spot data (5-min intervals)
- Download India VIX data
- Calculate realized volatility (5, 10, 20, 30-day windows)
- Handle missing data and market holidays
- Save to Parquet format

**Usage:**
```python
from backtesting import HistoricalDataCollector

collector = HistoricalDataCollector(kite=None)  # Pass Kite instance if available
data = collector.collect_and_save_data(
    start_date='2023-01-01',
    end_date='2023-12-31',
    interval='5minute'
)
```

### 3. `option_simulator.py`
Black-Scholes option pricing with Greeks.

**Features:**
- European call/put pricing
- Delta, Gamma, Theta, Vega, Rho calculation
- Volatility smile/skew application
- Bid-ask spread simulation

**Usage:**
```python
from backtesting import BlackScholesModel, OptionChainSimulator

bs_model = BlackScholesModel()
simulator = OptionChainSimulator(bs_model)

chain = simulator.generate_option_chain(
    spot=48000,
    atm_iv=0.18,
    time_to_expiry=3/365
)
```

### 4. `backtest_engine.py`
Main backtesting engine that simulates trading.

**Features:**
- Minute-by-minute simulation
- Entry signal generation
- Position management
- Exit condition checking
- P&L tracking

**Usage:**
```python
from backtesting import BacktestEngine

engine = BacktestEngine(market_data, strategy_config)
results = engine.run()

# Results contain:
# - trades: List of all trades
# - equity_curve: DataFrame with equity over time
# - final_capital: Ending capital
```

### 5. `backtest_analytics.py`
Performance metrics and report generation.

**Features:**
- Calculate Sharpe, Sortino, Calmar ratios
- Drawdown analysis
- Trade statistics
- Monthly returns heatmap
- HTML report generation

**Usage:**
```python
from backtesting import analyze_backtest_results

metrics = analyze_backtest_results(
    trades_df=trades_df,
    equity_curve=equity_curve,
    initial_capital=300000,
    output_dir='results/',
    strategy_config=config
)
```

### 6. `monte_carlo.py`
Monte Carlo simulations for robustness testing.

**Features:**
- Parameter variation testing
- Market regime testing
- Stress testing
- Bootstrap analysis

**Usage:**
```python
from backtesting import run_full_monte_carlo

mc_results = run_full_monte_carlo(
    market_data=data,
    strategy_config=config,
    output_dir='results/'
)
```

## ⚙️ Configuration

### Strategy Configuration

Edit `backtesting/backtest_config.py`:

```python
STRATEGY_CONFIG = {
    # Entry conditions
    'vix_range': (12, 25),          # VIX must be in this range
    'iv_percentile_min': 50,        # Min IV percentile to enter

    # Strike selection
    'strike_selection_method': 'DELTA',
    'ce_delta_target': -0.30,       # Target delta for CE
    'pe_delta_target': 0.30,        # Target delta for PE

    # Exit conditions
    'profit_target': 0.50,          # 50% profit target
    'stop_loss': 2.00,              # 200% stop loss
    'delta_threshold': 15,          # Exit if delta > ±15

    # Position sizing
    'initial_capital': 300000,
    'position_size': 1,             # Number of lots
    'lot_size': 15,                 # BankNifty lot size
}
```

### Transaction Costs

```python
COST_CONFIG = {
    'brokerage_per_lot': 40,        # ₹40 per lot round trip
    'stt_rate': 0.000125,           # 0.0125% on sell side
    'entry_slippage': 2,            # ₹2 slippage on entry
    'exit_slippage': 3,             # ₹3 slippage on exit
    'stop_loss_slippage': 5,        # ₹5 slippage on stop loss
}
```

### Option Pricing

```python
OPTION_PRICING_CONFIG = {
    'risk_free_rate': 0.06,         # 6% annual
    'skew_factor_puts': -0.15,      # OTM puts +15% IV
    'skew_factor_calls': -0.10,     # OTM calls +10% IV
    'spread_atm': 2.0,              # ₹2 ATM spread
}
```

## 📊 Usage Examples

### Example 1: Basic Backtest for 2023

```python
from backtesting import HistoricalDataCollector, BacktestEngine
from backtesting import analyze_backtest_results

# Collect data
collector = HistoricalDataCollector()
data = collector.load_from_parquet('combined_market_data.parquet')

# Filter to 2023
data_2023 = data['2023-01-01':'2023-12-31']

# Run backtest
engine = BacktestEngine(data_2023)
results = engine.run()

# Analyze
metrics = analyze_backtest_results(
    trades_df=engine.get_trades_dataframe(),
    equity_curve=results['equity_curve'],
    initial_capital=300000,
    output_dir='results/2023_backtest',
    strategy_config=results['strategy_config']
)

print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
```

### Example 2: Parameter Optimization

```python
from backtesting import BacktestEngine, STRATEGY_CONFIG
import pandas as pd

# Test different profit targets
profit_targets = [0.30, 0.40, 0.50, 0.60, 0.70]
results = []

for pt in profit_targets:
    config = STRATEGY_CONFIG.copy()
    config['profit_target'] = pt

    engine = BacktestEngine(data, config)
    backtest_results = engine.run()

    results.append({
        'profit_target': pt,
        'total_return': backtest_results['total_return'],
        'num_trades': len(engine.get_trades_dataframe())
    })

results_df = pd.DataFrame(results)
print(results_df)
```

### Example 3: Monte Carlo Simulation

```python
from backtesting import run_full_monte_carlo

mc_results = run_full_monte_carlo(
    market_data=data,
    strategy_config=STRATEGY_CONFIG,
    output_dir='results/monte_carlo'
)

# View bootstrap confidence intervals
ci = mc_results['bootstrap']['confidence_intervals']
print(f"Expected Return (95% CI): "
      f"[{ci['total_return_pct']['95%']['lower']:.2f}%, "
      f"{ci['total_return_pct']['95%']['upper']:.2f}%]")
```

## 📁 Output Files

After running a backtest, the following files are generated:

```
data/backtest_results/backtest_YYYYMMDD_HHMMSS/
├── trades.csv                      # All trades with details
├── equity_curve.csv                # Equity over time
├── performance_metrics.json        # All calculated metrics
├── backtest_report.html            # Interactive HTML report
└── monte_carlo_results.json        # Monte Carlo results (if run)
```

### trades.csv
Contains all executed trades:
- Entry/exit times and prices
- Strikes
- P&L and P&L%
- Exit reason
- Hold time
- Transaction costs

### backtest_report.html
Interactive HTML report with:
- Equity curve chart
- Drawdown chart
- Monthly returns heatmap
- P&L distribution
- Trade duration distribution
- Performance metrics summary

## 🔬 Advanced Features

### Testing Specific Market Periods

```python
# COVID crash period
covid_data = data['2020-03-01':'2020-04-30']
engine = BacktestEngine(covid_data)
results = engine.run()

# Bull market 2021
bull_data = data['2021-01-01':'2021-12-31']
engine = BacktestEngine(bull_data)
results = engine.run()
```

### Custom Strategy Variations

```python
# More aggressive settings
aggressive_config = STRATEGY_CONFIG.copy()
aggressive_config['profit_target'] = 0.30
aggressive_config['stop_loss'] = 3.00
aggressive_config['position_size'] = 2

# More conservative settings
conservative_config = STRATEGY_CONFIG.copy()
conservative_config['profit_target'] = 0.70
conservative_config['stop_loss'] = 1.50
conservative_config['vix_range'] = (15, 20)
```

### Regime Analysis

Compare performance across different market conditions:

```python
from backtesting import MonteCarloSimulator

simulator = MonteCarloSimulator(data, STRATEGY_CONFIG)
regime_results = simulator.run_regime_tests()

# Results for trending, ranging, high-vol, crash scenarios
for regime, metrics in regime_results.items():
    print(f"{regime}: Return = {metrics['metrics']['total_return_pct']:.2f}%")
```

## 📈 Interpreting Results

### Key Metrics to Watch

1. **Total Return & CAGR**: Overall profitability
2. **Sharpe Ratio**: Risk-adjusted returns (>1 is good, >2 is excellent)
3. **Max Drawdown**: Largest peak-to-trough decline
4. **Win Rate**: Should be >60% for short premium strategies
5. **Profit Factor**: Total wins / Total losses (>1.5 is good)
6. **Average Hold Time**: Typical trade duration

### Red Flags

- ⚠️ Win rate <50%
- ⚠️ Sharpe ratio <0.5
- ⚠️ Max drawdown >30%
- ⚠️ Profit factor <1.0
- ⚠️ Very few trades (<10)

### Validation

- Test on multiple time periods
- Run Monte Carlo simulations
- Check performance across different VIX regimes
- Validate against live trading results

## 🛠️ Troubleshooting

### Common Issues

**Issue: No trades generated**
- Check VIX range in config
- Check IV percentile threshold
- Verify data has sufficient history
- Check for market holidays

**Issue: Unrealistic returns**
- Verify transaction costs are enabled
- Check slippage settings
- Review option pricing skew

**Issue: Memory errors**
- Reduce data range
- Use daily data instead of 5-minute
- Disable detailed logging

## 📝 Notes

### Assumptions

1. **European Options**: Uses Black-Scholes (weekly options are European-style)
2. **No Early Assignment**: Weekly options rarely assigned early
3. **Liquidity**: Assumes sufficient liquidity for fills
4. **No Dividends**: BankNifty index has no dividend yield
5. **Market Impact**: Not modeled (small position sizes)

### Limitations

1. **Historical Data**: Past performance doesn't guarantee future results
2. **Option Pricing**: Black-Scholes may not perfectly match real market prices
3. **Events**: Major event filtering is manual (update event dates in config)
4. **Execution**: Assumes all orders fill at expected prices
5. **Margin**: Margin requirements not explicitly modeled

## 🤝 Contributing

To add new features:

1. Add configuration to `backtest_config.py`
2. Implement feature in appropriate module
3. Update this README
4. Test thoroughly

## 📄 License

This backtesting framework is part of the BankNifty Options Trading System.

## 📧 Support

For issues or questions:
1. Check this README
2. Review configuration files
3. Check log files for errors
4. Open an issue on GitHub

---

**Disclaimer**: This backtesting framework is for educational and research purposes only. Options trading involves substantial risk. Past performance does not guarantee future results. Always perform your own due diligence before trading with real money.
