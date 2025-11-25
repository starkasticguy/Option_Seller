#!/usr/bin/env python3
"""
BankNifty Options Backtesting - Main Script
Orchestrates the entire backtesting process
"""

import sys
import os
import argparse
from datetime import datetime
import logging

# Add backtesting directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backtesting'))

from backtesting.backtest_config import (
    DATA_CONFIG, STRATEGY_CONFIG, BACKTEST_CONFIG,
    validate_backtest_config, create_directories,
    get_backtest_config_summary
)
from backtesting.data_collector import HistoricalDataCollector
from backtesting.backtest_engine import BacktestEngine
from backtesting.backtest_analytics import analyze_backtest_results
from backtesting.monte_carlo import run_full_monte_carlo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     BANKNIFTY OPTIONS BACKTESTING FRAMEWORK              ║
    ║                                                          ║
    ║     Comprehensive Strategy Testing & Optimization        ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='BankNifty Options Backtesting Framework'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=DATA_CONFIG['start_date'],
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=DATA_CONFIG['end_date'],
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--interval',
        type=str,
        default='5minute',
        choices=['5minute', '15minute', '30minute', 'day'],
        help='Data interval'
    )

    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect historical data before backtesting'
    )

    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip main backtest (only collect data)'
    )

    parser.add_argument(
        '--run-monte-carlo',
        action='store_true',
        help='Run Monte Carlo simulations'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default=STRATEGY_CONFIG['strategy_type'],
        choices=['SHORT_STRANGLE', 'LONG_STRANGLE', 'IRON_CONDOR'],
        help='Strategy type'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=STRATEGY_CONFIG['initial_capital'],
        help='Initial capital'
    )

    parser.add_argument(
        '--position-size',
        type=int,
        default=STRATEGY_CONFIG['position_size'],
        help='Position size (number of lots)'
    )

    parser.add_argument(
        '--profit-target',
        type=float,
        default=STRATEGY_CONFIG['profit_target'],
        help='Profit target (as decimal, e.g., 0.5 for 50%%)'
    )

    parser.add_argument(
        '--stop-loss',
        type=float,
        default=STRATEGY_CONFIG['stop_loss'],
        help='Stop loss (as decimal, e.g., 2.0 for 200%%)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )

    parser.add_argument(
        '--load-data',
        action='store_true',
        help='Load existing data instead of collecting new'
    )

    return parser.parse_args()


def collect_data(args):
    """Collect historical data"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Data Collection")
    logger.info("=" * 60)

    collector = HistoricalDataCollector(kite=None)

    if args.load_data:
        logger.info("Loading existing data...")
        try:
            combined_data = collector.load_from_parquet('combined_market_data.parquet')
            logger.info(f"✅ Loaded {len(combined_data)} records")
            return combined_data
        except FileNotFoundError:
            logger.warning("No existing data found. Collecting new data...")

    # Collect new data
    combined_data = collector.collect_and_save_data(
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval
    )

    return combined_data


def run_backtest(market_data, args):
    """Run main backtest"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Running Backtest")
    logger.info("=" * 60)

    # Update strategy config with command line arguments
    strategy_config = STRATEGY_CONFIG.copy()
    strategy_config['strategy_type'] = args.strategy
    strategy_config['initial_capital'] = args.initial_capital
    strategy_config['position_size'] = args.position_size
    strategy_config['profit_target'] = args.profit_target
    strategy_config['stop_loss'] = args.stop_loss

    logger.info("\nStrategy Configuration:")
    logger.info(f"  Type: {strategy_config['strategy_type']}")
    logger.info(f"  Initial Capital: ₹{strategy_config['initial_capital']:,.0f}")
    logger.info(f"  Position Size: {strategy_config['position_size']} lot(s)")
    logger.info(f"  Profit Target: {strategy_config['profit_target']*100}%")
    logger.info(f"  Stop Loss: {strategy_config['stop_loss']*100}%")

    # Initialize and run backtest engine
    engine = BacktestEngine(market_data, strategy_config)
    results = engine.run()

    return results, strategy_config


def analyze_results(results, strategy_config, args):
    """Analyze backtest results"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Analyzing Results")
    logger.info("=" * 60)

    # Get trades and equity curve
    trades_df = results['trades']
    equity_curve = results['equity_curve']
    initial_capital = results['initial_capital']

    # Convert trades to DataFrame if it's a list
    if isinstance(trades_df, list):
        import pandas as pd
        if trades_df:
            from backtesting.backtest_engine import BacktestEngine
            # Create a temporary engine to get the DataFrame conversion
            temp_df_data = []
            for trade in trades_df:
                temp_df_data.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'position_type': trade.position_type,
                    'ce_strike': trade.ce_strike,
                    'pe_strike': trade.pe_strike,
                    'ce_entry_price': trade.ce_entry_price,
                    'pe_entry_price': trade.pe_entry_price,
                    'ce_exit_price': trade.ce_exit_price,
                    'pe_exit_price': trade.pe_exit_price,
                    'lots': trade.lots,
                    'pnl': trade.pnl,
                    'pnl_percent': trade.pnl_percent,
                    'exit_reason': trade.exit_reason,
                    'hold_time_minutes': trade.hold_time_minutes,
                    'transaction_costs': trade.transaction_costs,
                })
            trades_df = pd.DataFrame(temp_df_data)
        else:
            trades_df = pd.DataFrame()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"{DATA_CONFIG['backtest_results_dir']}/backtest_{timestamp}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Analyze results
    metrics = analyze_backtest_results(
        trades_df=trades_df,
        equity_curve=equity_curve,
        initial_capital=initial_capital,
        output_dir=output_dir,
        strategy_config=strategy_config
    )

    # Print summary
    print_summary(metrics)

    return metrics, output_dir


def run_monte_carlo_analysis(market_data, strategy_config, output_dir):
    """Run Monte Carlo simulations"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Monte Carlo Simulations")
    logger.info("=" * 60)

    mc_results = run_full_monte_carlo(
        market_data=market_data,
        strategy_config=strategy_config,
        output_dir=output_dir
    )

    # Print Monte Carlo summary
    if 'bootstrap' in mc_results and mc_results['bootstrap']:
        ci = mc_results['bootstrap'].get('confidence_intervals', {})
        if ci:
            logger.info("\nBootstrap Confidence Intervals (95%):")
            for metric, intervals in ci.items():
                if '95%' in intervals:
                    interval_95 = intervals['95%']
                    logger.info(f"  {metric}:")
                    logger.info(f"    Mean: {interval_95['mean']:.2f}")
                    logger.info(f"    95% CI: [{interval_95['lower']:.2f}, {interval_95['upper']:.2f}]")

    return mc_results


def print_summary(metrics):
    """Print summary of results"""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  Initial Capital:     ₹{metrics.get('initial_capital', 0):,.2f}")
    print(f"  Final Capital:       ₹{metrics.get('final_capital', 0):,.2f}")
    print(f"  Total Return:        {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  CAGR:                {metrics.get('cagr', 0):.2f}%")

    print(f"\n📈 RISK METRICS:")
    print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Volatility (Annual): {metrics.get('volatility_annual_pct', 0):.2f}%")

    print(f"\n💰 TRADE STATISTICS:")
    print(f"  Total Trades:        {metrics.get('num_trades', 0)}")
    print(f"  Winning Trades:      {metrics.get('num_winning_trades', 0)}")
    print(f"  Losing Trades:       {metrics.get('num_losing_trades', 0)}")
    print(f"  Win Rate:            {metrics.get('win_rate_pct', 0):.2f}%")
    print(f"  Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
    print(f"  Avg Trade P&L:       ₹{metrics.get('avg_trade_pnl', 0):,.2f}")

    print("\n" + "=" * 60)


def main():
    """Main execution function"""
    # Print banner
    print_banner()

    # Parse arguments
    args = parse_arguments()

    # Validate configuration
    errors = validate_backtest_config()
    if errors:
        logger.error("Configuration errors found:")
        for error in errors:
            logger.error(f"  ❌ {error}")
        return

    # Create directories
    create_directories()

    # Print configuration
    logger.info(get_backtest_config_summary())

    try:
        # Step 1: Collect or load data
        if args.collect_data or not args.load_data:
            market_data = collect_data(args)
        else:
            logger.info("Loading existing data...")
            collector = HistoricalDataCollector(kite=None)
            market_data = collector.load_from_parquet('combined_market_data.parquet')

        if args.skip_backtest:
            logger.info("✅ Data collection completed. Skipping backtest.")
            return

        # Step 2: Run backtest
        results, strategy_config = run_backtest(market_data, args)

        # Step 3: Analyze results
        metrics, output_dir = analyze_results(results, strategy_config, args)

        # Step 4: Monte Carlo (optional)
        if args.run_monte_carlo:
            mc_results = run_monte_carlo_analysis(market_data, strategy_config, output_dir)

        # Final message
        logger.info("\n" + "=" * 60)
        logger.info("✅ BACKTEST COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info("=" * 60 + "\n")

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Backtest interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n\n❌ Error during backtesting: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
