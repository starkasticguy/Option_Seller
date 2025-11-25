"""
BankNifty Options Backtesting Framework

A comprehensive backtesting framework for BankNifty options strategies
including:
- Historical data collection and processing
- Black-Scholes option pricing and Greeks calculation
- Strategy backtesting engine
- Performance analytics and reporting
- Monte Carlo simulations

Usage:
    from backtesting import BacktestEngine, HistoricalDataCollector

    # Collect data
    collector = HistoricalDataCollector()
    data = collector.collect_and_save_data('2020-01-01', '2024-12-31')

    # Run backtest
    engine = BacktestEngine(data)
    results = engine.run()

    # Analyze results
    from backtesting import analyze_backtest_results
    metrics = analyze_backtest_results(
        results['trades'],
        results['equity_curve'],
        results['initial_capital'],
        'output_dir'
    )
"""

__version__ = '1.0.0'
__author__ = 'BankNifty Trading Team'

# Import main classes for easy access
from backtesting.backtest_config import (
    STRATEGY_CONFIG,
    COST_CONFIG,
    DATA_CONFIG,
    OPTION_PRICING_CONFIG,
    MONTE_CARLO_CONFIG,
    validate_backtest_config,
    create_directories,
)

from backtesting.data_collector import HistoricalDataCollector

from backtesting.option_simulator import (
    BlackScholesModel,
    OptionChainSimulator,
    VolatilityCalculator,
)

from backtesting.backtest_engine import (
    BacktestEngine,
    Position,
    Trade,
)

from backtesting.backtest_analytics import (
    PerformanceAnalytics,
    ReportGenerator,
    analyze_backtest_results,
)

from backtesting.monte_carlo import (
    MonteCarloSimulator,
    run_full_monte_carlo,
)

__all__ = [
    # Configuration
    'STRATEGY_CONFIG',
    'COST_CONFIG',
    'DATA_CONFIG',
    'OPTION_PRICING_CONFIG',
    'MONTE_CARLO_CONFIG',
    'validate_backtest_config',
    'create_directories',

    # Data collection
    'HistoricalDataCollector',

    # Option pricing
    'BlackScholesModel',
    'OptionChainSimulator',
    'VolatilityCalculator',

    # Backtesting
    'BacktestEngine',
    'Position',
    'Trade',

    # Analytics
    'PerformanceAnalytics',
    'ReportGenerator',
    'analyze_backtest_results',

    # Monte Carlo
    'MonteCarloSimulator',
    'run_full_monte_carlo',
]
