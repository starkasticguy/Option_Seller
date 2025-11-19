"""
Backtesting Configuration Module
Contains all parameters and settings for the backtesting framework
"""

from datetime import datetime
from typing import Dict, List, Tuple
import os

# ========================
# DATA SETTINGS
# ========================
DATA_CONFIG = {
    # Data paths
    'data_dir': 'data',
    'historical_data_dir': 'data/historical',
    'backtest_results_dir': 'data/backtest_results',
    'cache_dir': 'data/cache',

    # Data sources
    'spot_data_file': 'data/historical/banknifty_spot.parquet',
    'vix_data_file': 'data/historical/india_vix.parquet',
    'combined_data_file': 'data/historical/combined_market_data.parquet',

    # Data collection parameters
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'interval': '5minute',  # 5-minute candles
    'instruments': {
        'spot': 'NSE:NIFTY BANK',
        'vix': 'INDIA VIX'
    },

    # Data quality
    'fill_missing_data': True,
    'interpolation_method': 'linear',
    'handle_holidays': True,
}

# ========================
# OPTION PRICING SETTINGS
# ========================
OPTION_PRICING_CONFIG = {
    # Black-Scholes parameters
    'risk_free_rate': 0.06,  # 6% annual risk-free rate (typical for India)
    'dividend_yield': 0.0,   # BankNifty has no dividend

    # Volatility parameters
    'use_realized_vol': True,
    'realized_vol_windows': [5, 10, 20, 30],  # Days for RV calculation
    'default_rv_window': 20,  # Default window for pricing

    # Volatility smile/skew parameters
    'apply_skew': True,
    'skew_factor_puts': -0.15,  # OTM puts have higher IV
    'skew_factor_calls': -0.10,  # OTM calls have slightly higher IV
    'skew_reference': 'atm',     # Reference point for skew calculation

    # Bid-ask spread simulation
    'simulate_spreads': True,
    'spread_atm': 2.0,          # ₹2 spread for ATM options
    'spread_otm_factor': 1.5,   # OTM spreads are 1.5x ATM
    'spread_deep_otm_factor': 2.5,  # Deep OTM spreads are 2.5x ATM
    'otm_threshold': 0.20,      # 20% OTM is considered "deep OTM"

    # Option chain generation
    'strike_interval': 100,     # Strike interval for BankNifty
    'strikes_range': 1500,      # Generate strikes ± 1500 from spot
    'expiry_day': 'Wednesday',  # Weekly expiry day
}

# ========================
# STRATEGY PARAMETERS
# ========================
STRATEGY_CONFIG = {
    # Strategy type
    'strategy_type': 'SHORT_STRANGLE',  # SHORT_STRANGLE, LONG_STRANGLE, IRON_CONDOR

    # Capital and position sizing
    'initial_capital': 300000,
    'position_size': 1,  # Number of lots
    'lot_size': 15,      # BankNifty lot size
    'max_positions': 1,  # Maximum simultaneous positions

    # Entry conditions
    'vix_range': (12, 25),           # VIX must be in this range
    'iv_percentile_min': 50,         # Minimum IV percentile to enter
    'min_dte': 0,                    # Minimum days to expiry
    'max_dte': 7,                    # Maximum days to expiry (weekly)

    # Strike selection
    'strike_selection_method': 'DELTA',  # DELTA, ATM_OFFSET, PREMIUM
    'ce_delta_target': -0.30,            # Target delta for CE
    'pe_delta_target': 0.30,             # Target delta for PE
    'atm_offset': 200,                   # ATM offset in points
    'min_premium': 50,                   # Minimum premium per option
    'max_premium': 300,                  # Maximum premium per option

    # Exit conditions
    'profit_target': 0.50,      # 50% of premium collected
    'stop_loss': 2.00,          # 200% of premium (2x loss)
    'delta_threshold': 15,      # Exit if portfolio delta exceeds ±15
    'exit_on_expiry_day': True, # Exit on expiry day
    'expiry_exit_time': '14:00', # Exit time on expiry day

    # Position management
    'trail_stop_enabled': False,
    'trail_stop_trigger': 0.40,
    'trail_stop_distance': 0.20,
    'rebalance_delta': False,
    'delta_rebalance_threshold': 20,

    # Time filters
    'no_trade_first_minutes': 15,  # No entry in first 15 mins
    'no_trade_last_minutes': 15,   # No entry in last 15 mins
    'trading_hours': ('09:20', '15:15'),  # IST trading window
}

# ========================
# TRANSACTION COSTS
# ========================
COST_CONFIG = {
    # Brokerage and fees
    'brokerage_per_lot': 40,        # ₹40 per lot (round trip)
    'brokerage_type': 'flat',       # 'flat' or 'percentage'
    'brokerage_percentage': 0.0003, # 0.03% (if using percentage)

    # Statutory charges
    'stt_rate': 0.000125,           # 0.0125% on sell side (options)
    'stt_on_buy': False,            # STT only on sell for options
    'exchange_charges': 0.00005,    # 0.005% exchange transaction charge
    'gst_rate': 0.18,               # 18% GST on brokerage
    'sebi_charges': 0.0000001,      # ₹10 per crore
    'stamp_duty': 0.00003,          # 0.003% on buy side

    # Slippage
    'slippage_enabled': True,
    'entry_slippage': 2,            # ₹2 slippage on entry
    'exit_slippage': 3,             # ₹3 slippage on normal exit
    'stop_loss_slippage': 5,        # ₹5 slippage on stop loss

    # Total impact calculation
    'use_detailed_costs': True,     # Calculate all costs separately
}

# ========================
# MARKET EVENTS
# ========================
EVENTS_CONFIG = {
    # Skip trading around major events
    'skip_major_events': True,
    'event_horizon_hours': 24,

    # Major event dates (add specific dates here)
    'major_events': [
        # RBI Policy Dates (examples - update with actual dates)
        '2020-03-27', '2020-05-22', '2020-08-06', '2020-10-09', '2020-12-04',
        '2021-02-05', '2021-04-07', '2021-06-04', '2021-08-06', '2021-10-08', '2021-12-08',
        '2022-02-10', '2022-04-08', '2022-06-08', '2022-08-05', '2022-09-30', '2022-12-07',
        '2023-02-08', '2023-04-06', '2023-06-08', '2023-08-10', '2023-10-06', '2023-12-08',
        '2024-02-08', '2024-04-05', '2024-06-07', '2024-08-08', '2024-10-09', '2024-12-06',

        # Budget dates
        '2020-02-01', '2021-02-01', '2022-02-01', '2023-02-01', '2024-02-01',

        # Major FOMC dates (examples)
        '2020-03-18', '2020-09-16', '2020-12-16',
        '2021-03-17', '2021-06-16', '2021-09-22', '2021-12-15',
        '2022-03-16', '2022-05-04', '2022-06-15', '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14',
        '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14', '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
        '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
    ],

    # Market holidays (examples - update with actual holidays)
    'market_holidays': [
        # 2020
        '2020-01-26', '2020-03-10', '2020-04-02', '2020-04-06', '2020-04-10',
        '2020-04-14', '2020-05-01', '2020-05-25', '2020-10-02', '2020-11-16',
        '2020-11-30', '2020-12-25',
        # 2021
        '2021-01-26', '2021-03-11', '2021-03-29', '2021-04-02', '2021-04-14',
        '2021-04-21', '2021-05-13', '2021-07-21', '2021-08-19', '2021-09-10',
        '2021-10-15', '2021-11-04', '2021-11-05', '2021-11-19',
        # 2022
        '2022-01-26', '2022-03-01', '2022-03-18', '2022-04-14', '2022-04-15',
        '2022-05-03', '2022-08-09', '2022-08-15', '2022-08-31', '2022-10-05',
        '2022-10-24', '2022-10-26', '2022-11-08',
        # 2023
        '2023-01-26', '2023-03-07', '2023-03-30', '2023-04-04', '2023-04-07',
        '2023-04-14', '2023-05-01', '2023-06-28', '2023-08-15', '2023-09-19',
        '2023-10-02', '2023-10-24', '2023-11-14', '2023-11-27', '2023-12-25',
        # 2024
        '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11',
        '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17',
        '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25',
    ],
}

# ========================
# BACKTESTING ENGINE
# ========================
BACKTEST_CONFIG = {
    # Simulation parameters
    'simulation_mode': 'tick',      # 'tick' or 'bar' level simulation
    'warmup_days': 30,              # Days of warmup for indicators
    'max_lookback': 60,             # Maximum lookback period

    # Execution simulation
    'execution_delay': 0,           # Bars delay for execution (0 = immediate)
    'use_realistic_fills': True,    # Use bid/ask for fills
    'fill_at_limit': 0.5,          # Probability of limit order fill

    # Performance tracking
    'track_greeks': True,
    'track_iv': True,
    'calculate_metrics_frequency': 'daily',  # 'tick', 'daily', 'weekly'

    # Output settings
    'save_trades': True,
    'save_equity_curve': True,
    'save_detailed_log': True,
    'generate_report': True,

    # Logging
    'log_level': 'INFO',
    'log_signals': True,
    'log_fills': True,
    'log_errors': True,
}

# ========================
# MONTE CARLO SETTINGS
# ========================
MONTE_CARLO_CONFIG = {
    # Simulation parameters
    'num_simulations': 1000,
    'num_paths': 100,
    'confidence_levels': [0.05, 0.25, 0.50, 0.75, 0.95],

    # Parameter variations
    'vary_parameters': True,
    'param_variations': {
        'vix_range': [
            (10, 20), (12, 25), (15, 30)
        ],
        'profit_target': [0.30, 0.50, 0.70],
        'stop_loss': [1.50, 2.00, 2.50],
        'delta_target': [0.25, 0.30, 0.35],
    },

    # Market regime testing
    'test_regimes': True,
    'regimes': {
        'trending_bull': {'drift': 0.0003, 'volatility_mult': 0.8},
        'trending_bear': {'drift': -0.0003, 'volatility_mult': 1.2},
        'ranging': {'drift': 0, 'volatility_mult': 0.9},
        'high_vol': {'drift': 0, 'volatility_mult': 1.5},
        'crash': {'drift': -0.002, 'volatility_mult': 2.5},
    },

    # Stress testing
    'stress_test': True,
    'stress_scenarios': {
        'sudden_gap': {'gap_percent': 0.03, 'frequency': 0.02},  # 3% gap, 2% probability
        'vol_spike': {'vol_mult': 2.0, 'frequency': 0.05},       # 2x vol, 5% probability
        'circuit_filter': {'move_percent': 0.10},                 # 10% move triggers circuit
    },
}

# ========================
# ANALYTICS SETTINGS
# ========================
ANALYTICS_CONFIG = {
    # Performance metrics
    'calculate_sharpe': True,
    'calculate_sortino': True,
    'calculate_calmar': True,
    'calculate_omega': True,
    'risk_free_rate': 0.06,         # For Sharpe ratio calculation

    # Drawdown analysis
    'track_drawdowns': True,
    'max_dd_periods': 'all',        # Track all drawdown periods

    # Trade analysis
    'analyze_by_regime': True,
    'analyze_by_expiry': True,
    'analyze_by_vix': True,
    'vix_buckets': [0, 12, 15, 18, 21, 25, 100],

    # Returns analysis
    'returns_period': 'daily',      # 'daily', 'weekly', 'monthly'
    'compound_returns': True,

    # Visualization
    'create_plots': True,
    'plot_equity_curve': True,
    'plot_drawdown': True,
    'plot_monthly_returns': True,
    'plot_returns_dist': True,
    'plot_underwater': True,

    # Report generation
    'html_report': True,
    'pdf_report': False,
    'include_charts': True,
    'chart_style': 'plotly',        # 'plotly' or 'matplotlib'
}

# ========================
# OPTIMIZATION SETTINGS
# ========================
OPTIMIZATION_CONFIG = {
    # Walk-forward analysis
    'walk_forward_enabled': False,
    'train_period_months': 12,
    'test_period_months': 3,
    'reoptimize_frequency': 'quarterly',

    # Parameter optimization
    'optimize_params': False,
    'optimization_method': 'grid',  # 'grid', 'random', 'bayesian'
    'objective_function': 'sharpe_ratio',  # or 'total_return', 'calmar'

    # Grid search parameters
    'param_grid': {
        'profit_target': [0.30, 0.40, 0.50, 0.60, 0.70],
        'stop_loss': [1.50, 2.00, 2.50, 3.00],
        'delta_target': [0.20, 0.25, 0.30, 0.35, 0.40],
        'vix_min': [10, 12, 14],
        'vix_max': [20, 25, 30],
    },
}

# ========================
# VALIDATION FUNCTIONS
# ========================
def validate_backtest_config() -> List[str]:
    """
    Validate backtesting configuration

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate dates
    try:
        start = datetime.strptime(DATA_CONFIG['start_date'], '%Y-%m-%d')
        end = datetime.strptime(DATA_CONFIG['end_date'], '%Y-%m-%d')
        if start >= end:
            errors.append("Start date must be before end date")
    except ValueError:
        errors.append("Invalid date format (use YYYY-MM-DD)")

    # Validate capital
    if STRATEGY_CONFIG['initial_capital'] < 100000:
        errors.append("Initial capital too low (minimum 100,000)")

    # Validate VIX range
    vix_low, vix_high = STRATEGY_CONFIG['vix_range']
    if vix_low >= vix_high:
        errors.append("Invalid VIX range")

    # Validate profit/loss targets
    if STRATEGY_CONFIG['profit_target'] <= 0:
        errors.append("Profit target must be positive")
    if STRATEGY_CONFIG['stop_loss'] <= 0:
        errors.append("Stop loss must be positive")

    # Validate lot size
    if STRATEGY_CONFIG['lot_size'] <= 0:
        errors.append("Lot size must be positive")

    # Validate delta targets
    ce_delta = abs(STRATEGY_CONFIG['ce_delta_target'])
    pe_delta = abs(STRATEGY_CONFIG['pe_delta_target'])
    if ce_delta > 1 or pe_delta > 1:
        errors.append("Delta targets must be between -1 and 1")

    return errors

def create_directories() -> None:
    """Create necessary directories for backtesting"""
    directories = [
        DATA_CONFIG['data_dir'],
        DATA_CONFIG['historical_data_dir'],
        DATA_CONFIG['backtest_results_dir'],
        DATA_CONFIG['cache_dir'],
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_backtest_config_summary() -> str:
    """
    Get a summary of backtesting configuration

    Returns:
        Formatted configuration summary
    """
    summary = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║          BACKTESTING CONFIGURATION SUMMARY               ║
    ╚══════════════════════════════════════════════════════════╝

    DATA PERIOD:
      Start Date: {DATA_CONFIG['start_date']}
      End Date: {DATA_CONFIG['end_date']}
      Interval: {DATA_CONFIG['interval']}

    STRATEGY:
      Type: {STRATEGY_CONFIG['strategy_type']}
      Initial Capital: ₹{STRATEGY_CONFIG['initial_capital']:,}
      Position Size: {STRATEGY_CONFIG['position_size']} lot(s)
      Lot Size: {STRATEGY_CONFIG['lot_size']} units

    ENTRY CONDITIONS:
      VIX Range: {STRATEGY_CONFIG['vix_range'][0]} - {STRATEGY_CONFIG['vix_range'][1]}
      IV Percentile Min: {STRATEGY_CONFIG['iv_percentile_min']}%
      Strike Selection: {STRATEGY_CONFIG['strike_selection_method']}
      CE Delta Target: {STRATEGY_CONFIG['ce_delta_target']}
      PE Delta Target: {STRATEGY_CONFIG['pe_delta_target']}

    EXIT CONDITIONS:
      Profit Target: {STRATEGY_CONFIG['profit_target']*100}%
      Stop Loss: {STRATEGY_CONFIG['stop_loss']*100}%
      Delta Threshold: ±{STRATEGY_CONFIG['delta_threshold']}

    TRANSACTION COSTS:
      Brokerage: ₹{COST_CONFIG['brokerage_per_lot']} per lot
      STT: {COST_CONFIG['stt_rate']*100}%
      Entry Slippage: ₹{COST_CONFIG['entry_slippage']}
      Exit Slippage: ₹{COST_CONFIG['exit_slippage']}

    MONTE CARLO:
      Simulations: {MONTE_CARLO_CONFIG['num_simulations']}
      Vary Parameters: {MONTE_CARLO_CONFIG['vary_parameters']}
      Test Regimes: {MONTE_CARLO_CONFIG['test_regimes']}
    """
    return summary

# ========================
# HELPER FUNCTIONS
# ========================
def get_cost_per_trade(premium: float, lots: int) -> Dict[str, float]:
    """
    Calculate all costs for a trade

    Args:
        premium: Total premium for the trade
        lots: Number of lots

    Returns:
        Dictionary with cost breakdown
    """
    lot_size = STRATEGY_CONFIG['lot_size']
    turnover = premium * lots * lot_size

    # Brokerage
    if COST_CONFIG['brokerage_type'] == 'flat':
        brokerage = COST_CONFIG['brokerage_per_lot'] * lots
    else:
        brokerage = turnover * COST_CONFIG['brokerage_percentage']

    # STT (only on sell side for options)
    stt = turnover * COST_CONFIG['stt_rate'] if not COST_CONFIG['stt_on_buy'] else 0

    # Exchange charges
    exchange = turnover * COST_CONFIG['exchange_charges']

    # GST on brokerage
    gst = brokerage * COST_CONFIG['gst_rate']

    # SEBI charges
    sebi = turnover * COST_CONFIG['sebi_charges']

    # Stamp duty (on buy side)
    stamp = turnover * COST_CONFIG['stamp_duty']

    total_cost = brokerage + stt + exchange + gst + sebi + stamp

    return {
        'brokerage': brokerage,
        'stt': stt,
        'exchange_charges': exchange,
        'gst': gst,
        'sebi_charges': sebi,
        'stamp_duty': stamp,
        'total_cost': total_cost,
        'cost_percentage': (total_cost / turnover * 100) if turnover > 0 else 0
    }

if __name__ == "__main__":
    # Validate configuration
    errors = validate_backtest_config()

    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("✅ Configuration validated successfully!")
        print(get_backtest_config_summary())

    # Create directories
    create_directories()
    print("\n✅ Directories created successfully!")
