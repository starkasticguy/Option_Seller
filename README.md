# BankNifty Options Trading System

A comprehensive real-time options trading system for BankNifty weekly options using Zerodha's KiteConnect API. The system implements delta-neutral strangles with dynamic hedging, risk management, and economic event monitoring.

## Features

- **Real-time Options Trading**: Live option chain data and automated signal generation
- **Delta-Neutral Strategy**: Short/long strangles with automatic delta hedging
- **Risk Management**: Comprehensive position sizing, VaR, stop-loss, and profit targets
- **Economic Event Monitoring**: Tracks high-impact events and adjusts trading accordingly
- **VIX-Based Trading**: Only trades within optimal VIX ranges
- **Real-time Console Display**: Beautiful colored console output with live updates
- **Trade Logging**: SQLite database for complete trade history and analytics
- **Paper Trading Mode**: Test strategies without risking real capital
- **Performance Analytics**: Win rate, profit factor, Sharpe ratio, and more

## Project Structure

```
banknifty_trader/
├── config.py              # Configuration and parameters
├── executor.py            # Main trading loop and display
├── strategy.py            # Core trading strategy logic
├── market_monitor.py      # Event detection and VIX monitoring
├── risk_manager.py        # Risk management and position sizing
├── database.py            # Trade logging and history
├── utils.py               # Utility functions
└── trades.db             # SQLite database (created automatically)
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

### 3. Configure API Credentials

Copy the example environment file and add your Zerodha credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
ZERODHA_USER_ID=your_user_id_here
ZERODHA_PASSWORD=your_password_here
ZERODHA_PIN=your_pin_here
```

### 4. Update Configuration

Edit `banknifty_trader/config.py` to customize trading parameters:

```python
TRADING_PARAMS = {
    'capital': 300000,              # Your trading capital
    'max_risk_per_trade': 0.02,     # 2% risk per trade
    'position_size': 1,             # Number of lots
    'vix_range': (12, 25),          # Acceptable VIX range
    'delta_threshold': 15,          # Max delta before hedging
    'profit_target': 0.5,           # 50% of premium
    'stop_loss': 2.0,               # 200% of premium
}
```

## Getting Started with Zerodha

### 1. Create API App

1. Go to [https://developers.kite.trade/](https://developers.kite.trade/)
2. Login with your Zerodha credentials
3. Create a new app
4. Note down your API Key and API Secret

### 2. Set Redirect URL

Set the redirect URL to: `http://localhost:8080`

### 3. Get Access Token

The system will automatically handle the login flow the first time you run it. Follow the prompts to authorize the application.

## Usage

### Paper Trading (Recommended for Testing)

Run the system in paper trading mode (no real orders):

```bash
cd banknifty_trader
python executor.py
```

OR

```bash
python -m banknifty_trader.executor
```

### Live Trading

**⚠️ WARNING: This will place real orders!**

```bash
python executor.py --live
```

### Console Output

The system provides a real-time console display that updates every 5 seconds:

```
============================================================
    BANKNIFTY OPTIONS TRADING SYSTEM - PAPER TRADING
============================================================
Session Start: 2024-01-15 09:15:30
Current Time: 10:30:45

📊 MARKET STATUS:
BankNifty: 48,235.50
VIX: 15.60 (Normal) 🟢
Event Risk: GREEN 🟢
Message: No major events - NORMAL TRADING
Trading: ✅ ALLOWED

📈 CURRENT POSITION:
Type: SHORT_STRANGLE
Strikes: CE 48500, PE 47900
Entry: CE ₹150.25, PE ₹145.50
Entry Time: 09:30:00
Duration: 60 minutes

P&L: 📈 ₹1,245.50 (+42.1%)
Peak Profit: ₹1,450.00
Max Loss: ₹-350.00
Target: ₹2,220.00
Stop: ₹-5,910.00
Delta: +2.50

🎯 TRADING SIGNAL:
⏸️  ACTION: HOLD POSITION
   Current P&L: ₹1,245.50
   Delta: +2.50
   Reason: Position healthy - P&L: +42.1%
   Confidence: 70.0%

⚠️  RISK METRICS:
Daily P&L: ₹1,245.50 (+0.42%)
Daily Limit: ₹6,000
Remaining: ₹4,754.50
VaR (95%): ₹3,200
Positions: 1/1
Trades Today: 1

📅 UPCOMING EVENTS (Next 24hrs):
   ✅ No high impact events

============================================================
Refreshing in 5 seconds... Press Ctrl+C to stop
============================================================
```

## Configuration Guide

### Trading Parameters

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| `capital` | Total trading capital | 300,000 | Min 100,000 |
| `max_risk_per_trade` | Risk per trade (%) | 0.02 | 1-3% |
| `position_size` | Number of lots | 1 | Start with 1 |
| `vix_range` | Acceptable VIX range | (12, 25) | Avoid extremes |
| `delta_threshold` | Max delta before hedge | 15 | 10-20 |
| `profit_target` | Profit target (% of premium) | 0.5 | 40-60% |
| `stop_loss` | Stop loss (% of premium) | 2.0 | 150-250% |

### Strategy Types

1. **SHORT_STRANGLE** (Default)
   - Sell OTM call and put options
   - Profit from time decay and low volatility
   - Best when IV is high (>60th percentile)

2. **LONG_STRANGLE**
   - Buy OTM call and put options
   - Profit from large moves
   - Best when expecting volatility expansion

### Strike Selection Methods

1. **DELTA**: Select strikes based on delta targets (e.g., 0.30 delta)
2. **ATM_OFFSET**: Select strikes at fixed distance from ATM
3. **PREMIUM**: Select based on premium range

## Risk Management

The system includes multiple layers of risk protection:

### Position Limits
- Maximum positions open simultaneously
- Maximum daily loss limit
- Consecutive loss protection with cooldown

### Greek Limits
- Delta: ±15 (adjustable)
- Gamma: Max 1000
- Vega: Max 5000
- Theta: Min -2000/day

### Emergency Exits
- Daily loss > ₹6,000
- VIX spike > 20%
- Underlying move > 2% in 5 minutes

## Database

All trades are logged to SQLite database (`trades.db`) with the following tables:

- **trades**: Complete trade records
- **signals**: All trading signals generated
- **greeks_history**: Greek values over time
- **market_data**: Market condition snapshots
- **daily_performance**: Daily statistics

### Query Trade History

```python
from banknifty_trader.database import TradingDatabase

db = TradingDatabase()

# Get last 30 days of trades
trades = db.get_trade_history(days=30)

# Get today's statistics
stats = db.get_daily_stats()

# Get overall performance
summary = db.get_performance_summary()
```

## Performance Analytics

The system calculates and displays:

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / gross losses
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Average profit vs average loss

## Economic Event Monitoring

The system monitors and reacts to:

- **High Impact Events** (Score 4-5): Skip trading
  - RBI Monetary Policy
  - US FOMC Meeting
  - Major CPI/GDP releases

- **Medium Impact Events** (Score 3): Reduce position size
  - Trade balance
  - Industrial production
  - Retail sales

## Troubleshooting

### Issue: "KiteConnect not installed"

```bash
pip install kiteconnect
```

### Issue: "API credentials not configured"

Edit `config.py` or `.env` file with your Zerodha credentials.

### Issue: "Access token expired"

Delete the old access token and re-authenticate.

### Issue: "Database locked"

Close any other programs accessing the database.

### Issue: "Module not found"

Make sure you're in the correct directory:

```bash
cd banknifty_trader
python executor.py
```

## Important Notes

### Capital Requirements

- Minimum recommended capital: ₹100,000
- BankNifty lot size: 15
- Keep sufficient margin for hedging

### Trading Hours

- Market open: 09:15 AM IST
- System starts trading: 09:20 AM IST (after market settles)
- Last entry: 03:00 PM IST
- Exit all positions: 03:15 PM IST

### Commissions and Costs

- Update commission costs in config
- Default: ₹40 per lot (round trip)
- Consider STT, exchange fees, GST

### Market Holidays

The system does not automatically check for market holidays. Do not run on:
- NSE holidays
- Muhurat trading days
- Settlement days

## Advanced Features

### Custom Event Calendar

Add known events to `market_monitor.py`:

```python
known_events = [
    ('2024-02-08 14:00', 'RBI Monetary Policy', 5),
    ('2024-02-15 20:00', 'US FOMC Meeting', 5),
]
```

### Backtesting

```python
from banknifty_trader.strategy import BankNiftyOptionsTrader
from config import BACKTEST_CONFIG

# Enable backtesting in config
BACKTEST_CONFIG['enabled'] = True
BACKTEST_CONFIG['start_date'] = '2023-01-01'
BACKTEST_CONFIG['end_date'] = '2023-12-31'

# Run backtest
# (Implementation pending)
```

### Telegram Notifications

Configure in `config.py`:

```python
NOTIFICATION_CONFIG = {
    'telegram_enabled': True,
    'telegram_bot_token': 'your_bot_token',
    'telegram_chat_id': 'your_chat_id',
}
```

## Safety Guidelines

1. **Start with Paper Trading**: Test thoroughly before going live
2. **Small Position Sizes**: Start with 1 lot
3. **Set Daily Limits**: Never risk more than you can afford
4. **Monitor Regularly**: Don't leave the system unattended
5. **Understand the Code**: Review all modules before using
6. **Keep Backups**: Back up your database regularly
7. **Update Regularly**: Keep dependencies and API libraries updated

## Disclaimer

**⚠️ IMPORTANT DISCLAIMER:**

This software is provided for educational purposes only. Trading in options involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.

- Use at your own risk
- The authors are not responsible for any financial losses
- Always test in paper trading mode first
- Consult with a financial advisor before trading
- Options trading can result in total loss of capital

## License

This project is for educational and research purposes. See LICENSE file for details.

## Support

For issues, questions, or contributions:

1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue on GitHub
4. Consult Zerodha API documentation

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit a pull request

## Changelog

### Version 1.0.0 (2024-01-15)

- Initial release
- Basic short strangle strategy
- Risk management system
- Economic event monitoring
- Real-time console display
- SQLite trade logging
- Paper trading mode

## Roadmap

- [ ] Backtesting module
- [ ] Telegram bot integration
- [ ] Web dashboard
- [ ] Iron Condor strategy
- [ ] Machine learning signal enhancement
- [ ] Multi-timeframe analysis
- [ ] Options chain heatmap
- [ ] Alert system for large moves

## Acknowledgments

- Zerodha for KiteConnect API
- The Python trading community
- Options pricing models and research

---

**Happy Trading! 📈**

Remember: The best trade is often the one you don't take. Trade responsibly!
