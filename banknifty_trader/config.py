"""
Configuration file for BankNifty Options Trading System
Store your API credentials and trading parameters here
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ========================
# ZERODHA API CREDENTIALS
# ========================
# For manual login flow:
# 1. Set ZERODHA_API_KEY and ZERODHA_API_SECRET in .env
# 2. Run the system, it will generate a login URL
# 3. Login manually and copy the request token
# 4. System will generate an access token
# 5. Save ZERODHA_ACCESS_TOKEN in .env for future use
# Note: Access tokens are valid until midnight
ZERODHA_CONFIG = {
    'api_key': os.getenv('ZERODHA_API_KEY', 'YOUR_API_KEY'),
    'api_secret': os.getenv('ZERODHA_API_SECRET', 'YOUR_API_SECRET'),
    'user_id': os.getenv('ZERODHA_USER_ID', 'YOUR_USER_ID'),  # Not needed for API login
    'password': os.getenv('ZERODHA_PASSWORD', 'YOUR_PASSWORD'),  # Not needed for API login
    'pin': os.getenv('ZERODHA_PIN', 'YOUR_PIN'),  # Not needed for API login
    'request_token': None,  # Provided manually during login
    'access_token': os.getenv('ZERODHA_ACCESS_TOKEN')  # Auto-loaded from .env if available
}

# ========================
# TRADING PARAMETERS
# ========================
TRADING_PARAMS = {
    # Capital and Risk Management
    'capital': 300000,              # Total trading capital
    'max_risk_per_trade': 0.02,     # 2% risk per trade
    'max_daily_loss': 6000,         # Maximum daily loss in rupees
    'max_portfolio_loss': 0.05,     # 5% max portfolio loss before shutdown

    # Position Sizing
    'position_size': 1,             # Number of lots to trade
    'max_positions': 1,             # Maximum simultaneous positions

    # Volatility Parameters
    'vix_range': (12, 25),          # Acceptable VIX range for trading
    'vix_threshold_high': 25,       # Stop trading if VIX above this
    'vix_threshold_low': 12,        # Don't trade if VIX too low
    'iv_percentile_min': 50,        # Minimum IV percentile to sell options

    # Delta Management
    'delta_threshold': 15,          # Max portfolio delta before hedging
    'target_delta': 0,              # Target neutral delta
    'delta_adjustment_size': 0.25,  # Futures lot size for hedging

    # Profit/Loss Targets
    'profit_target': 0.5,           # 50% of premium collected
    'stop_loss': 2.0,               # 200% of premium (2x loss)
    'trail_stop_trigger': 0.4,      # Start trailing at 40% profit
    'trail_stop_distance': 0.2,     # Trail by 20% from peak

    # Time Management
    'trading_hours': ('09:20', '15:15'),  # IST trading window
    'no_trade_before': '09:20',           # Wait for market to settle
    'exit_before_close': '15:15',         # Exit all positions before close
    'days_to_expiry_min': 0,              # Minimum DTE for new positions
    'days_to_expiry_max': 7,              # Maximum DTE (weekly options)
    'exit_on_expiry_day': True,           # Auto-exit on expiry day

    # Strategy Specific
    'strategy_type': 'SHORT_STRANGLE',    # SHORT_STRANGLE, LONG_STRANGLE, IRON_CONDOR
    'strike_selection': 'DELTA',          # DELTA, ATM_OFFSET, PREMIUM
    'ce_delta_target': -0.30,             # Target delta for CE strike
    'pe_delta_target': 0.30,              # Target delta for PE strike
    'atm_offset': 200,                    # Points away from ATM (if using ATM_OFFSET)
    'min_premium': 50,                    # Minimum premium per lot
    'max_premium': 300,                   # Maximum premium per lot

    # Hedging
    'use_hedging': False,                 # Enable futures hedging
    'hedge_instrument': 'BANKNIFTY-FUT',  # Future or far OTM options
    'hedge_ratio': 0.25,                  # Hedge ratio (0.25 = 25% of position)
}

# ========================
# INSTRUMENT SETTINGS
# ========================
INSTRUMENT_CONFIG = {
    'symbol': 'BANKNIFTY',
    'exchange': 'NFO',
    'lot_size': 15,                     # BankNifty lot size (verify current)
    'tick_size': 0.05,
    'expiry_day': 'Wednesday',          # Weekly expiry day
    'segment': 'NFO-OPT',
}

# ========================
# MARKET DATA SETTINGS
# ========================
MARKET_DATA = {
    'india_vix_symbol': 'INDIA VIX',
    'banknifty_index': 'NSE:NIFTY BANK',
    'nifty_index': 'NSE:NIFTY 50',
    'refresh_interval': 5,               # Seconds between updates
    'option_chain_refresh': 30,          # Seconds between option chain refresh
}

# ========================
# EVENT MONITORING
# ========================
EVENTS_CONFIG = {
    # Economic Calendar APIs
    'economic_calendar_api': 'https://api.investing.com/api/financialdata/calendar',
    'india_events_url': 'https://www.moneycontrol.com/economy-finance/ecocalendar',
    'forex_factory_url': 'https://www.forexfactory.com/calendar',

    # Event Impact Scores (0-5)
    'event_scores': {
        'RBI Monetary Policy': 5,
        'US FOMC Meeting': 5,
        'India CPI': 4,
        'US NFP': 4,
        'India GDP': 4,
        'US CPI': 4,
        'India Trade Balance': 3,
        'US GDP': 4,
        'Major Bank Earnings': 3,
        'India IIP': 3,
        'US Retail Sales': 3,
    },

    # Risk levels based on events
    'event_horizon_hours': 24,          # Check events in next 24 hours
    'high_impact_threshold': 4,         # Score 4+ = high impact
    'medium_impact_threshold': 3,       # Score 3 = medium impact

    # Actions based on event risk
    'skip_trading_on_high_impact': True,
    'reduce_size_on_medium_impact': True,
    'size_reduction_factor': 0.5,       # Reduce to 50% on medium impact
}

# ========================
# RISK MANAGEMENT RULES
# ========================
RISK_RULES = {
    # Circuit Breakers
    'daily_loss_limit': 6000,           # Stop trading if daily loss exceeds
    'max_consecutive_losses': 3,        # Stop after 3 consecutive losses
    'cooldown_period_minutes': 60,      # Wait period after max losses

    # Position Limits
    'max_margin_usage': 0.50,           # Use max 50% of available margin
    'emergency_exit_loss': 0.03,        # Exit all if portfolio down 3%

    # Volatility Filters
    'max_sudden_vix_spike': 0.20,       # Exit if VIX spikes >20% suddenly
    'max_underlying_move': 0.02,        # Monitor if BankNifty moves >2% in 5min

    # Greek Limits
    'max_portfolio_gamma': 1000,        # Maximum gamma exposure
    'max_portfolio_vega': 5000,         # Maximum vega exposure
    'max_portfolio_theta': -2000,       # Maximum theta decay per day
}

# ========================
# DATABASE SETTINGS
# ========================
DATABASE_CONFIG = {
    'db_path': 'banknifty_trader/trades.db',
    'log_all_ticks': False,             # Log every tick (heavy I/O)
    'log_signals': True,                # Log all trading signals
    'log_greeks': True,                 # Log greek calculations
    'backup_interval_hours': 24,        # Backup database every 24h
}

# ========================
# LOGGING SETTINGS
# ========================
LOGGING_CONFIG = {
    'log_level': 'INFO',                # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,
    'log_file_path': 'banknifty_trader/logs/trading.log',
    'log_rotation': '100 MB',           # Rotate log file at 100MB
    'log_retention': '30 days',         # Keep logs for 30 days
    'console_output': True,             # Print to console
}

# ========================
# NOTIFICATION SETTINGS
# ========================
NOTIFICATION_CONFIG = {
    'telegram_enabled': False,
    'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),

    # Email notifications
    'email_enabled': False,
    'email_smtp_server': 'smtp.gmail.com',
    'email_smtp_port': 587,
    'email_from': os.getenv('EMAIL_FROM', ''),
    'email_password': os.getenv('EMAIL_PASSWORD', ''),
    'email_to': os.getenv('EMAIL_TO', ''),

    # Alert conditions
    'notify_on_entry': True,
    'notify_on_exit': True,
    'notify_on_error': True,
    'notify_daily_summary': True,
}

# ========================
# DISPLAY SETTINGS
# ========================
DISPLAY_CONFIG = {
    'update_interval': 5,               # Console refresh rate (seconds)
    'use_colors': True,                 # Enable colored output
    'clear_screen': True,               # Clear screen on each update
    'show_greeks': True,
    'show_events': True,
    'show_risk_metrics': True,
    'decimal_places': 2,
}

# ========================
# BACKTESTING SETTINGS
# ========================
BACKTEST_CONFIG = {
    'enabled': False,
    'start_date': '2023-01-01',
    'end_date': '2024-12-31',
    'initial_capital': 300000,
    'commission_per_lot': 40,           # Total round-trip commission
    'slippage_points': 5,               # Average slippage in points
}

# ========================
# VALIDATION
# ========================
def validate_config():
    """Validate configuration parameters"""
    errors = []

    # Check API credentials
    if ZERODHA_CONFIG['api_key'] == 'YOUR_API_KEY':
        errors.append("Zerodha API key not configured")

    # Check capital
    if TRADING_PARAMS['capital'] < 100000:
        errors.append("Capital too low for options trading (min 100000)")

    # Check VIX range
    vix_low, vix_high = TRADING_PARAMS['vix_range']
    if vix_low >= vix_high:
        errors.append("Invalid VIX range")

    # Check profit/loss ratios
    if TRADING_PARAMS['profit_target'] <= 0 or TRADING_PARAMS['stop_loss'] <= 0:
        errors.append("Invalid profit target or stop loss")

    return errors

def get_config_summary():
    """Print configuration summary"""
    summary = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║     BANKNIFTY OPTIONS TRADING SYSTEM - CONFIGURATION     ║
    ╚══════════════════════════════════════════════════════════╝

    Capital: ₹{TRADING_PARAMS['capital']:,}
    Max Daily Loss: ₹{TRADING_PARAMS['max_daily_loss']:,}
    Position Size: {TRADING_PARAMS['position_size']} lot(s)
    Strategy: {TRADING_PARAMS['strategy_type']}

    VIX Range: {TRADING_PARAMS['vix_range'][0]} - {TRADING_PARAMS['vix_range'][1]}
    Delta Threshold: ±{TRADING_PARAMS['delta_threshold']}

    Profit Target: {TRADING_PARAMS['profit_target']*100}% of premium
    Stop Loss: {TRADING_PARAMS['stop_loss']*100}% of premium

    Trading Hours: {TRADING_PARAMS['trading_hours'][0]} - {TRADING_PARAMS['trading_hours'][1]} IST
    """
    return summary

if __name__ == "__main__":
    # Validate configuration
    errors = validate_config()
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validated successfully!")
        print(get_config_summary())
