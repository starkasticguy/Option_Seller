# BankNifty Options Data Collection System

A real-time market data collection and analysis system for BankNifty weekly options using Zerodha's KiteConnect API. This system collects option chain data, Greeks, volatility metrics, and other market data for analysis.

**Note:** This system is focused on **data collection and analysis only**. Trading/execution functionality will be added in the future.

## Features

- **Real-time Market Data Collection**: Live BankNifty spot price and India VIX
- **Option Chain Data**: Complete option chain snapshots with bid/ask/LTP
- **Volatility Metrics**: Historical volatility (HV), implied volatility (IV), IV percentile
- **Greeks Tracking**: Delta, Gamma, Theta, Vega for all strikes
- **Open Interest & Volume**: Track OI and volume changes
- **Put-Call Ratio (PCR)**: OI and volume-based PCR
- **SQLite Database**: All data stored locally for analysis
- **Real-time Console Display**: Beautiful colored console output with live updates
- **Mock Data Mode**: Test without Zerodha API connection

## Project Structure

```
banknifty_trader/
├── config.py              # Configuration and parameters
├── data_collector.py      # Main data collection loop and display
├── data_analyzer.py       # Option chain and volatility analysis
├── market_monitor.py      # VIX and volatility monitoring
├── database.py            # Market data logging
├── utils.py               # Utility functions
└── market_data.db         # SQLite database (created automatically)
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Option_Seller
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials (Optional)

The system can run in mock data mode without API credentials. To collect real market data:

Copy the example environment file and add your Zerodha credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your **API Key and Secret**:

```env
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
```

### 4. Update Configuration (Optional)

Edit `banknifty_trader/config.py` to customize data collection parameters:

```python
DATA_COLLECTION = {
    'market_data_interval': 5,          # Seconds between market data snapshots
    'option_chain_interval': 30,        # Seconds between option chain updates
    'volatility_calc_interval': 60,     # Seconds between volatility calculations
}
```

## Getting Started with Zerodha (Optional)

### 1. Create API App

1. Go to [https://developers.kite.trade/](https://developers.kite.trade/)
2. Login with your Zerodha credentials
3. Click "Create New App"
4. Fill in the app details:
   - App Name: BankNifty Data Collector
   - Redirect URL: `http://127.0.0.1`
   - Description: Options data collection system
5. Note down your **API Key** and **API Secret**

### 2. Manual Login Flow (First Time)

When you run the system for the first time with API credentials, you'll need to manually login:

1. **System generates login URL**: Copy the URL displayed in the terminal
2. **Login manually**: Open the URL in your browser and login
3. **Get request token**: Copy the request_token from the redirected URL
4. **Paste request token**: Paste it into the terminal
5. **Save access token**: The system will display an access token - save it in your `.env` file

### 3. Reusing Access Tokens

Access tokens are valid until midnight. To avoid manual login every time:

1. After first login, copy the access token displayed by the system
2. Add it to your `.env` file: `ZERODHA_ACCESS_TOKEN=...`
3. The system will automatically use this token on subsequent runs
4. Generate a new token daily (tokens expire at midnight IST)

## Usage

### Start Data Collection

```bash
python run.py
```

OR

```bash
cd banknifty_trader
python data_collector.py
```

### Console Output

The system provides a real-time console display that updates every 5 seconds:

```
======================================================================
    BANKNIFTY OPTIONS DATA COLLECTION SYSTEM
======================================================================
Session Start: 2024-01-15 09:15:30
Current Time: 10:30:45
Snapshots Collected: 125

📊 MARKET DATA:
BankNifty Spot: 48,235.50
India VIX: 15.60 (Normal) 🟢
  Change: -0.25 (-1.6%)
Historical Volatility: 18.50% (Normal)
  Daily Vol: 1.20% | Window: 20 days
Market Status: ✅ OPEN - COLLECTING DATA

📈 OPTION CHAIN DATA:
ATM Strike: 48,200
CE Premium: ₹150.25 | PE Premium: ₹145.50
CE IV: 18.50% | PE IV: 18.75%
Put-Call Ratio (OI): 1.05
Total CE OI: 1,250,000 | Total PE OI: 1,312,500
Last Snapshot: 15s ago

📊 GREEKS ANALYSIS:
ATM CE Greeks:
  Delta: 0.520 | Gamma: 0.0025
  Theta: -45.50 | Vega: 28.50
ATM PE Greeks:
  Delta: -0.480 | Gamma: 0.0025
  Theta: -44.20 | Vega: 27.80

📁 COLLECTION STATISTICS:
Session Duration: 75 minutes
Total Snapshots: 125
Collection Rate: 1.7 snapshots/min
DB Records:
  Market Data: 900
  Option Chain: 150
  Volatility: 75

======================================================================
Refreshing in 5 seconds... Press Ctrl+C to stop
======================================================================
```

## Configuration Guide

### Data Collection Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `market_data_interval` | Seconds between market snapshots | 5 |
| `option_chain_interval` | Seconds between option chain updates | 30 |
| `volatility_calc_interval` | Seconds between volatility calculations | 60 |
| `store_option_chain` | Store option chain snapshots | True |
| `store_greeks` | Store Greeks data | True |
| `store_volatility` | Store volatility metrics | True |

### Data Collection Hours

```python
COLLECTION_HOURS = {
    'start_time': '09:15',              # Market open
    'end_time': '15:30',                # Market close
}
```

## Database

All market data is logged to SQLite database (`market_data.db`) with the following tables:

- **market_data**: BankNifty spot, VIX, volatility snapshots
- **option_chain**: Complete option chain snapshots
- **volatility_data**: IV, HV, IV percentile metrics
- **pcr_data**: Put-Call Ratio data
- **greeks_data**: Aggregated Greeks metrics

### Query Data with Python

```python
from banknifty_trader.database import MarketDataDatabase

db = MarketDataDatabase()

# Get last 24 hours of market data
market_data = db.get_market_data(days=1)

# Get last hour of option chain snapshots
option_chain = db.get_option_chain_snapshots(hours=1)

# Get collection statistics
stats = db.get_collection_stats()
print(stats)
```

### Export to CSV for Analysis

```python
import pandas as pd
from banknifty_trader.database import MarketDataDatabase

db = MarketDataDatabase()

# Export market data
market_data = db.get_market_data(days=7)
market_data.to_csv('market_data_7days.csv', index=False)

# Export option chain
option_chain = db.get_option_chain_snapshots(hours=6)
option_chain.to_csv('option_chain_6hours.csv', index=False)
```

## Analysis Use Cases

With the collected data, you can:

1. **Volatility Analysis**
   - Compare IV vs HV
   - Track IV percentile over time
   - Identify volatility expansion/contraction

2. **Option Chain Analysis**
   - Identify support/resistance from OI
   - Track max pain levels
   - Analyze PCR for sentiment

3. **Greeks Analysis**
   - Track delta distribution
   - Monitor gamma exposure zones
   - Analyze theta decay patterns

4. **Strategy Backtesting** (Future)
   - Use collected data for backtesting
   - Build execution module later

## Troubleshooting

### Issue: "KiteConnect not installed"

```bash
pip install kiteconnect
```

### Issue: "API credentials not configured"

The system will run in mock data mode. To collect real data, configure API credentials in `.env` file.

### Issue: "Access token expired" or "TokenException"

Access tokens expire daily at midnight IST. To fix:

1. Remove the old `ZERODHA_ACCESS_TOKEN` from your `.env` file
2. Run the system again
3. Follow the manual login flow to get a new token
4. Save the new token in your `.env` file

### Issue: "Database locked"

Close any other programs accessing the database.

## Important Notes

### Data Collection Hours

- Market open: 09:15 AM IST
- System starts collecting: 09:15 AM IST
- Market close: 03:30 PM IST
- System stops: 03:30 PM IST

### Market Holidays

The system does not automatically check for market holidays. Do not run on:
- NSE holidays
- Muhurat trading days

### Disk Space

Option chain snapshots can consume significant disk space over time. Monitor your disk usage and clean up old data periodically.

## Roadmap

- [x] Real-time data collection
- [x] Option chain snapshots
- [x] Volatility tracking
- [x] Greeks calculation
- [x] SQLite storage
- [ ] Data export utilities
- [ ] Analysis notebooks
- [ ] Backtesting module
- [ ] Strategy development
- [ ] Execution module
- [ ] Web dashboard

## Disclaimer

**⚠️ IMPORTANT DISCLAIMER:**

This software is provided for educational and research purposes only. This is a data collection tool and does NOT execute any trades.

- Use at your own risk
- Market data may have delays or inaccuracies
- Always verify data before using in analysis
- This is not financial advice

## License

This project is for educational and research purposes. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit a pull request

## Changelog

### Version 2.0.0 (2025-01-01)

- **Complete refactor to data collection only**
- Removed all trading/execution functionality
- Added comprehensive market data logging
- Added volatility tracking
- Added PCR metrics
- Simplified codebase for data collection

### Version 1.0.0 (2024-01-15)

- Initial release with trading system
- (Now deprecated in favor of data collection)

---

**Happy Data Collecting! 📊**

Collect data first, analyze later, trade smart!
