"""
Configuration file for BankNifty Options Data Collection System
Collect market data and option chain information for analysis
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
    'request_token': None,  # Provided manually during login
    'access_token': os.getenv('ZERODHA_ACCESS_TOKEN')  # Auto-loaded from .env if available
}

# ========================
# DATA COLLECTION PARAMETERS
# ========================
DATA_COLLECTION = {
    # Data collection intervals
    'market_data_interval': 5,          # Seconds between market data snapshots
    'option_chain_interval': 30,        # Seconds between option chain updates
    'volatility_calc_interval': 60,     # Seconds between volatility calculations

    # Data storage
    'store_tick_data': False,           # Store every tick (heavy I/O)
    'store_option_chain': True,         # Store complete option chain snapshots
    'store_greeks': True,               # Store calculated greeks
    'store_volatility': True,           # Store volatility metrics

    # Historical data
    'historical_days': 60,              # Days of historical data to fetch
    'lookback_days': 20,                # Days for volatility calculations
}

# ========================
# INSTRUMENT SETTINGS
# ========================
INSTRUMENT_CONFIG = {
    'symbol': 'BANKNIFTY',
    'exchange': 'NFO',
    'lot_size': 15,                     # BankNifty lot size
    'tick_size': 0.05,
    'expiry_day': 'Wednesday',          # Weekly expiry day
    'segment': 'NFO-OPT',
    'index_symbol': 'NSE:NIFTY BANK',
    'vix_symbol': 'NSE:INDIA VIX',
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
    'strike_range': 1000,                # Points above/below spot to collect
}

# ========================
# DATABASE SETTINGS
# ========================
DATABASE_CONFIG = {
    'db_path': 'banknifty_trader/market_data.db',
    'log_all_ticks': False,             # Log every tick (heavy I/O)
    'log_option_chain': True,           # Log option chain data
    'log_greeks': True,                 # Log greek calculations
    'log_volatility': True,             # Log volatility metrics
    'backup_interval_hours': 24,        # Backup database every 24h
}

# ========================
# LOGGING SETTINGS
# ========================
LOGGING_CONFIG = {
    'log_level': 'INFO',                # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,
    'log_file_path': 'banknifty_trader/logs/data_collection.log',
    'log_rotation': '100 MB',           # Rotate log file at 100MB
    'log_retention': '30 days',         # Keep logs for 30 days
    'console_output': True,             # Print to console
}

# ========================
# DISPLAY SETTINGS
# ========================
DISPLAY_CONFIG = {
    'update_interval': 5,               # Console refresh rate (seconds)
    'use_colors': True,                 # Enable colored output
    'clear_screen': True,               # Clear screen on each update
    'show_greeks': True,
    'show_volatility': True,
    'show_option_chain': True,
    'decimal_places': 2,
}

# ========================
# ANALYSIS SETTINGS
# ========================
ANALYSIS_CONFIG = {
    # Volatility analysis
    'iv_percentile_window': 60,         # Days for IV percentile calculation
    'hv_window': 20,                    # Days for historical volatility

    # Option chain analysis
    'analyze_oi': True,                 # Analyze open interest
    'analyze_volume': True,             # Analyze volume patterns
    'analyze_pcr': True,                # Calculate put-call ratio

    # Greeks analysis
    'calculate_implied_greeks': True,   # Calculate greeks from prices
    'track_delta': True,                # Track delta changes
    'track_gamma': True,                # Track gamma exposure
    'track_theta': True,                # Track theta decay
}

# ========================
# DATA COLLECTION HOURS
# ========================
COLLECTION_HOURS = {
    'start_time': '09:15',              # Market open
    'end_time': '15:30',                # Market close
    'collect_pre_market': False,        # Collect pre-market data
    'collect_post_market': False,       # Collect post-market data
}

# ========================
# VALIDATION
# ========================
def validate_config():
    """Validate configuration parameters"""
    errors = []

    # Check API credentials
    if ZERODHA_CONFIG['api_key'] == 'YOUR_API_KEY':
        errors.append("Zerodha API key not configured in .env file")

    # Check intervals
    if DATA_COLLECTION['market_data_interval'] < 1:
        errors.append("Market data interval too low (min 1 second)")

    return errors

def get_config_summary():
    """Print configuration summary"""
    summary = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║   BANKNIFTY OPTIONS DATA COLLECTION - CONFIGURATION      ║
    ╚══════════════════════════════════════════════════════════╝

    Data Collection Intervals:
    - Market Data: Every {DATA_COLLECTION['market_data_interval']} seconds
    - Option Chain: Every {DATA_COLLECTION['option_chain_interval']} seconds
    - Volatility: Every {DATA_COLLECTION['volatility_calc_interval']} seconds

    Data Storage:
    - Option Chain: {'Yes' if DATA_COLLECTION['store_option_chain'] else 'No'}
    - Greeks: {'Yes' if DATA_COLLECTION['store_greeks'] else 'No'}
    - Volatility: {'Yes' if DATA_COLLECTION['store_volatility'] else 'No'}

    Collection Hours: {COLLECTION_HOURS['start_time']} - {COLLECTION_HOURS['end_time']} IST

    Database: {DATABASE_CONFIG['db_path']}
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
