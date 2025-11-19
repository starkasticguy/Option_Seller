"""
Database Module
Handles SQLite database operations for trade logging and history
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import json
import pytz
from colorama import Fore

from config import DATABASE_CONFIG


class TradingDatabase:
    """SQLite database for trade and signal logging"""

    def __init__(self, db_path: str = None):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DATABASE_CONFIG['db_path']
        self.ist = pytz.timezone('Asia/Kolkata')
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"{Fore.RED}Error connecting to database: {str(e)}")

    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            # Trades table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    trade_type TEXT,
                    ce_strike INTEGER,
                    pe_strike INTEGER,
                    ce_entry_premium REAL,
                    pe_entry_premium REAL,
                    ce_exit_premium REAL,
                    pe_exit_premium REAL,
                    lots INTEGER,
                    lot_size INTEGER,
                    entry_spot REAL,
                    exit_spot REAL,
                    entry_vix REAL,
                    exit_vix REAL,
                    pnl REAL,
                    pnl_percent REAL,
                    exit_reason TEXT,
                    entry_delta REAL,
                    exit_delta REAL,
                    max_profit REAL,
                    max_loss REAL,
                    duration_minutes INTEGER,
                    commission REAL,
                    net_pnl REAL,
                    notes TEXT
                )
            ''')

            # Signals table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    signal_type TEXT,
                    action TEXT,
                    ce_strike INTEGER,
                    pe_strike INTEGER,
                    spot_price REAL,
                    vix REAL,
                    iv_percentile REAL,
                    confidence REAL,
                    reason TEXT,
                    executed BOOLEAN,
                    signal_data TEXT
                )
            ''')

            # Greeks history table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS greeks_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    trade_id INTEGER,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    spot_price REAL,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                )
            ''')

            # Market data table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    banknifty_spot REAL,
                    india_vix REAL,
                    vix_change_pct REAL,
                    event_status TEXT,
                    event_risk_score REAL
                )
            ''')

            # Performance metrics table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE,
                    trades_count INTEGER,
                    winners INTEGER,
                    losers INTEGER,
                    total_pnl REAL,
                    total_commission REAL,
                    net_pnl REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL
                )
            ''')

            self.conn.commit()

        except Exception as e:
            print(f"{Fore.RED}Error creating tables: {str(e)}")

    def log_trade(self, trade_data: Dict) -> int:
        """
        Log a trade to the database

        Args:
            trade_data: Trade details dictionary

        Returns:
            Trade ID
        """
        try:
            self.cursor.execute('''
                INSERT INTO trades (
                    entry_time, exit_time, trade_type,
                    ce_strike, pe_strike,
                    ce_entry_premium, pe_entry_premium,
                    ce_exit_premium, pe_exit_premium,
                    lots, lot_size,
                    entry_spot, exit_spot,
                    entry_vix, exit_vix,
                    pnl, pnl_percent,
                    exit_reason,
                    entry_delta, exit_delta,
                    max_profit, max_loss,
                    duration_minutes,
                    commission, net_pnl,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('entry_time'),
                trade_data.get('exit_time'),
                trade_data.get('trade_type'),
                trade_data.get('ce_strike'),
                trade_data.get('pe_strike'),
                trade_data.get('ce_entry_premium'),
                trade_data.get('pe_entry_premium'),
                trade_data.get('ce_exit_premium'),
                trade_data.get('pe_exit_premium'),
                trade_data.get('lots'),
                trade_data.get('lot_size'),
                trade_data.get('entry_spot'),
                trade_data.get('exit_spot'),
                trade_data.get('entry_vix'),
                trade_data.get('exit_vix'),
                trade_data.get('pnl'),
                trade_data.get('pnl_percent'),
                trade_data.get('exit_reason'),
                trade_data.get('entry_delta'),
                trade_data.get('exit_delta'),
                trade_data.get('max_profit', 0),
                trade_data.get('max_loss', 0),
                trade_data.get('duration_minutes'),
                trade_data.get('commission', 0),
                trade_data.get('net_pnl'),
                trade_data.get('notes', '')
            ))

            self.conn.commit()
            return self.cursor.lastrowid

        except Exception as e:
            print(f"{Fore.RED}Error logging trade: {str(e)}")
            return -1

    def log_signal(self, signal_data: Dict) -> int:
        """
        Log a trading signal

        Args:
            signal_data: Signal details dictionary

        Returns:
            Signal ID
        """
        try:
            self.cursor.execute('''
                INSERT INTO signals (
                    timestamp, signal_type, action,
                    ce_strike, pe_strike,
                    spot_price, vix, iv_percentile,
                    confidence, reason, executed,
                    signal_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(self.ist),
                signal_data.get('type', 'UNKNOWN'),
                signal_data.get('action'),
                signal_data.get('strikes', {}).get('CE', {}).get('strike'),
                signal_data.get('strikes', {}).get('PE', {}).get('strike'),
                signal_data.get('spot_price'),
                signal_data.get('vix'),
                signal_data.get('iv_percentile'),
                signal_data.get('confidence'),
                signal_data.get('reason'),
                signal_data.get('executed', False),
                json.dumps(signal_data)
            ))

            self.conn.commit()
            return self.cursor.lastrowid

        except Exception as e:
            print(f"{Fore.RED}Error logging signal: {str(e)}")
            return -1

    def log_greeks(self, trade_id: int, greeks: Dict):
        """
        Log portfolio greeks

        Args:
            trade_id: Associated trade ID
            greeks: Greeks dictionary
        """
        try:
            self.cursor.execute('''
                INSERT INTO greeks_history (
                    timestamp, trade_id, delta, gamma, theta, vega, spot_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(self.ist),
                trade_id,
                greeks.get('delta'),
                greeks.get('gamma'),
                greeks.get('theta'),
                greeks.get('vega'),
                greeks.get('spot_price')
            ))

            self.conn.commit()

        except Exception as e:
            print(f"{Fore.RED}Error logging greeks: {str(e)}")

    def log_market_data(self, market_data: Dict):
        """
        Log market data snapshot

        Args:
            market_data: Market data dictionary
        """
        try:
            self.cursor.execute('''
                INSERT INTO market_data (
                    timestamp, banknifty_spot, india_vix,
                    vix_change_pct, event_status, event_risk_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(self.ist),
                market_data.get('spot'),
                market_data.get('vix'),
                market_data.get('vix_change_pct'),
                market_data.get('event_status'),
                market_data.get('event_risk_score')
            ))

            self.conn.commit()

        except Exception as e:
            print(f"{Fore.RED}Error logging market data: {str(e)}")

    def get_trade_history(self, days: int = 30) -> List[Dict]:
        """
        Get trade history

        Args:
            days: Number of days to look back

        Returns:
            List of trade dictionaries
        """
        try:
            cutoff_date = datetime.now(self.ist) - pd.Timedelta(days=days)

            self.cursor.execute('''
                SELECT * FROM trades
                WHERE entry_time >= ?
                ORDER BY entry_time DESC
            ''', (cutoff_date,))

            columns = [desc[0] for desc in self.cursor.description]
            rows = self.cursor.fetchall()

            trades = []
            for row in rows:
                trade = dict(zip(columns, row))
                trades.append(trade)

            return trades

        except Exception as e:
            print(f"{Fore.RED}Error fetching trade history: {str(e)}")
            return []

    def get_daily_stats(self, date: str = None) -> Dict:
        """
        Get statistics for a specific day

        Args:
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Statistics dictionary
        """
        try:
            if date is None:
                date = datetime.now(self.ist).strftime('%Y-%m-%d')

            self.cursor.execute('''
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losers,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_profit,
                    MIN(pnl) as max_loss
                FROM trades
                WHERE DATE(entry_time) = ?
            ''', (date,))

            row = self.cursor.fetchone()

            if row:
                return {
                    'date': date,
                    'total_trades': row[0] or 0,
                    'winners': row[1] or 0,
                    'losers': row[2] or 0,
                    'total_pnl': row[3] or 0,
                    'avg_pnl': row[4] or 0,
                    'max_profit': row[5] or 0,
                    'max_loss': row[6] or 0,
                    'win_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0
                }

            return {}

        except Exception as e:
            print(f"{Fore.RED}Error fetching daily stats: {str(e)}")
            return {}

    def get_performance_summary(self) -> Dict:
        """
        Get overall performance summary

        Returns:
            Performance summary dictionary
        """
        try:
            self.cursor.execute('''
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losers,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                    MAX(pnl) as max_profit,
                    MIN(pnl) as max_loss,
                    AVG(duration_minutes) as avg_duration
                FROM trades
            ''')

            row = self.cursor.fetchone()

            if row and row[0]:
                total_trades = row[0]
                winners = row[1] or 0
                losers = row[2] or 0
                total_pnl = row[3] or 0
                avg_win = row[5] or 0
                avg_loss = abs(row[6]) if row[6] else 0

                win_rate = (winners / total_trades) if total_trades > 0 else 0
                profit_factor = (winners * avg_win) / (losers * avg_loss) if (losers * avg_loss) > 0 else 0

                return {
                    'total_trades': total_trades,
                    'winners': winners,
                    'losers': losers,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl': row[4],
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'max_profit': row[7],
                    'max_loss': row[8],
                    'avg_duration_minutes': row[9]
                }

            return {}

        except Exception as e:
            print(f"{Fore.RED}Error fetching performance summary: {str(e)}")
            return {}

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


if __name__ == "__main__":
    import pandas as pd

    print("Testing Database...")
    db = TradingDatabase()

    # Test trade logging
    sample_trade = {
        'entry_time': datetime.now(),
        'exit_time': datetime.now(),
        'trade_type': 'SHORT_STRANGLE',
        'ce_strike': 48500,
        'pe_strike': 47500,
        'ce_entry_premium': 150,
        'pe_entry_premium': 140,
        'ce_exit_premium': 75,
        'pe_exit_premium': 70,
        'lots': 1,
        'lot_size': 15,
        'entry_spot': 48000,
        'exit_spot': 48050,
        'entry_vix': 16.5,
        'exit_vix': 16.2,
        'pnl': 2250,
        'pnl_percent': 0.48,
        'exit_reason': 'Profit target',
        'entry_delta': -2.5,
        'exit_delta': -1.8,
        'duration_minutes': 120,
        'commission': 100,
        'net_pnl': 2150
    }

    trade_id = db.log_trade(sample_trade)
    print(f"Logged trade with ID: {trade_id}")

    # Test stats
    stats = db.get_daily_stats()
    print(f"\nDaily Stats: {stats}")

    summary = db.get_performance_summary()
    print(f"\nPerformance Summary: {summary}")

    print("\nDatabase module working successfully!")
