# CLAUDE.md - AI Assistant Guidelines for Option_Seller

## Project Overview

Option_Seller is a Python-based trading analysis system for NIFTY options data using the Zerodha Kite API. It downloads option chain data and calculates implied volatility (IV) metrics for options trading analysis.

## Repository Structure

```
Option_Seller/
├── auth.py                    # Zerodha OAuth authentication module
├── data_download.py           # Option chain data fetching from Kite API
├── implied_volatility.py      # Black-Scholes IV calculation script
├── .env.example               # Environment configuration template
├── access_token.json          # Stored API token (auto-generated)
├── option_chain_data.csv      # Raw option chain data output
└── option_chain_with_iv.csv   # Enriched data with IV calculations
```

## Key Components

### auth.py - Authentication Module
- **Class:** `ZerodhaAuth`
- **Purpose:** Handles OAuth login and token management for Zerodha Kite API
- **Key Methods:**
  - `get_login_url()` - Generates manual login URL
  - `generate_session(request_token)` - Exchanges request token for access token
  - `save_token()` / `load_token()` - Token persistence to `access_token.json`
  - `get_kite_instance()` - Returns authenticated KiteConnect instance
- **Note:** Uses manual OAuth flow (not credential-based) for security

### data_download.py - Data Downloader
- **Class:** `OptionChainDownloader`
- **Purpose:** Fetches real-time NIFTY option chain data
- **Key Methods:**
  - `get_nifty_spot_price()` - Gets current NIFTY 50 spot price
  - `get_option_instruments(expiry_date)` - Gets available option contracts
  - `get_option_chain_data(expiry_date, strike_range)` - Main data collection
  - `format_option_chain(df)` - Formats as traditional CE/PE side-by-side
- **Default behavior:** Fetches nearest expiry, ±500 points from spot

### implied_volatility.py - IV Calculator
- **Purpose:** Calculates implied volatility using Black-Scholes model
- **Key Functions:**
  - `calculate_iv_for_row(row)` - Optimizes IV per option using Nelder-Mead
  - `calculate_moneyness(row)` - Computes S/K (calls) or K/S (puts)
  - `categorize_moneyness(moneyness)` - ITM/ATM/OTM classification
- **Parameters:**
  - Risk-free rate: 5.25%
  - Initial IV guess: 14%
  - Valid IV range: 1-500%

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3 |
| Broker API | Zerodha Kite Connect |
| Data Processing | pandas |
| Numerical Computing | numpy |
| Optimization | scipy (fmin - Nelder-Mead) |

## Development Guidelines

### Code Conventions
- Use Python's `logging` module (INFO/ERROR levels)
- Wrap critical operations in try-except with logging
- Follow single-responsibility principle for classes
- Include docstrings with parameter/return documentation
- Output analysis results to CSV files

### Error Handling Pattern
```python
try:
    # operation
except Exception as e:
    logging.error(f"Description: {e}")
    # graceful degradation or raise
```

### Authentication Flow
1. Call `auth.get_login_url()` to get OAuth URL
2. User logs in manually via browser
3. Extract `request_token` from redirect URL
4. Call `auth.generate_session(request_token)`
5. Token auto-saved to `access_token.json`
6. Use `auth.load_token()` on subsequent runs

### Data Pipeline
1. Initialize `OptionChainDownloader` with authenticated Kite instance
2. Fetch spot price → filter instruments → batch fetch quotes
3. Run `implied_volatility.py` to add IV calculations
4. Results saved to CSV for analysis

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required - Get from https://developers.kite.trade/
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret

# Optional - For notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Dependencies

```
kiteconnect    # Zerodha API client
pandas         # Data manipulation
numpy          # Numerical operations
scipy          # Scientific computing (optimization)
```

## Common Tasks

### Running Data Download
```python
from auth import ZerodhaAuth
from data_download import OptionChainDownloader

auth = ZerodhaAuth()
auth.load_token()
kite = auth.get_kite_instance()

downloader = OptionChainDownloader(kite)
df = downloader.get_option_chain_data(strike_range=500)
df.to_csv('option_chain_data.csv', index=False)
```

### Calculating IV
```bash
python implied_volatility.py
# Reads option_chain_data.csv, outputs option_chain_with_iv.csv
```

## Important Notes for AI Assistants

1. **Token Expiry:** Zerodha tokens expire daily at midnight IST. Check `generated_at` in `access_token.json`.

2. **API Limits:** Be mindful of Zerodha API rate limits when fetching data.

3. **IV Edge Cases:** IV calculation may fail for deep ITM/OTM options or very short-dated options. These return `None`.

4. **Data Freshness:** CSV files contain point-in-time snapshots. Always check timestamps.

5. **Black-Scholes Assumptions:** The IV calculator assumes European-style options and constant volatility. NIFTY options are European-style.

6. **Testing Changes:** Since this connects to live trading APIs, test changes carefully. Use mock data where possible.

7. **Credentials:** Never commit `.env` or `access_token.json` to version control (they're in `.gitignore`).
