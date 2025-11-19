# Quick Start Guide

Get up and running with the BankNifty Options Trading System in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Zerodha trading account with API access
- Minimum ₹100,000 capital (recommended)

## Installation (Linux/Mac)

```bash
# 1. Clone the repository
git clone <repository-url>
cd Option_Seller

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Activate virtual environment
source venv/bin/activate
```

## Installation (Windows)

```powershell
# 1. Clone the repository
git clone <repository-url>
cd Option_Seller

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Configuration

### 1. Get Zerodha API Credentials

1. Go to [https://developers.kite.trade/](https://developers.kite.trade/)
2. Login with your Zerodha credentials
3. Create a new app
4. Note your **API Key** and **API Secret**

### 2. Configure Credentials

Edit `.env` file:

```env
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
ZERODHA_USER_ID=your_user_id_here
ZERODHA_PASSWORD=your_password_here
ZERODHA_PIN=your_pin_here
```

### 3. Customize Trading Parameters (Optional)

Edit `banknifty_trader/config.py`:

```python
TRADING_PARAMS = {
    'capital': 300000,           # YOUR CAPITAL HERE
    'position_size': 1,          # Start with 1 lot
    'vix_range': (12, 25),       # VIX range
    'profit_target': 0.5,        # 50% profit target
    'stop_loss': 2.0,            # 200% stop loss
}
```

## Running the System

### Paper Trading (Recommended First)

```bash
python run.py
```

OR

```bash
python -m banknifty_trader.executor
```

### Live Trading (⚠️ Real Money!)

```bash
python run.py --live
```

## What You'll See

The system displays a real-time dashboard:

```
============================================================
    BANKNIFTY OPTIONS TRADING SYSTEM - PAPER TRADING
============================================================

📊 MARKET STATUS:
BankNifty: 48,235.50
VIX: 15.60 (Normal) 🟢
Event Risk: GREEN 🟢
Trading: ✅ ALLOWED

📈 CURRENT POSITION:
Status: No open positions

🎯 TRADING SIGNAL:
⏭️  ACTION: SKIP - WAIT
   Reason: Waiting for optimal entry conditions
   Confidence: 50.0%

⚠️  RISK METRICS:
Daily P&L: ₹0.00 (+0.00%)
Daily Limit: ₹6,000
Positions: 0/1

📅 UPCOMING EVENTS:
   ✅ No high impact events

============================================================
Refreshing in 5 seconds... Press Ctrl+C to stop
============================================================
```

## Understanding the Signals

### ✅ BUY/SELL STRANGLE
System found a good trading opportunity and will enter a position.

### ⏸️ HOLD
Current position is healthy, holding for profit target.

### 🛑 EXIT
Exiting position (profit target hit, stop loss, or risk event).

### ⏭️ SKIP
Conditions not favorable for trading (high VIX, events, etc.).

## Testing the System

### 1. Watch the Console
Let it run for a few hours in paper trading mode to see how it works.

### 2. Check Trade History
```python
from banknifty_trader.database import TradingDatabase

db = TradingDatabase()
trades = db.get_trade_history(days=7)
print(f"Trades: {len(trades)}")
```

### 3. Review Configuration
Make sure the parameters match your risk tolerance:
- Daily loss limit
- Position size
- Profit/loss targets

## Common Issues

### "KiteConnect not installed"
```bash
pip install kiteconnect
```

### "API credentials not configured"
Edit `.env` file with your actual Zerodha credentials.

### "Module not found"
Make sure you're in the correct directory and virtual environment is activated.

### "VIX too high/low"
The system won't trade if VIX is outside the configured range (12-25 by default).

## Safety Tips

1. ✅ **Start with paper trading** - Test for at least a week
2. ✅ **Use small position sizes** - Start with 1 lot
3. ✅ **Monitor regularly** - Don't leave unattended
4. ✅ **Set daily limits** - Stick to your risk tolerance
5. ✅ **Understand the code** - Review before using
6. ⚠️ **Never risk more than you can afford to lose**

## Next Steps

1. ✅ Run in paper trading mode for 1-2 weeks
2. ✅ Review all trades in the database
3. ✅ Adjust parameters based on your preference
4. ✅ Read the full README.md
5. ✅ Understand risk management features
6. ⚠️ Only then consider live trading

## Getting Help

- 📖 Read the full [README.md](README.md)
- 🔍 Check [Troubleshooting](README.md#troubleshooting) section
- 📊 Review your trade logs in `trades.db`
- 💬 Open an issue on GitHub

## Important Reminder

**This is for educational purposes only. Options trading involves substantial risk of loss. Always trade responsibly and within your means.**

---

Happy (Paper) Trading! 🎯
