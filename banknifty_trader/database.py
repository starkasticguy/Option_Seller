"""
Database Module
Handles SQLite database operations for market data logging
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import pytz
import pandas as pd
from colorama import Fore

from config import DATABASE_CONFIG


class MarketDataDatabase:
    """SQLite database for market data logging"""

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
        self._ensure_db_directory()
        self._connect()
        self._create_tables()

    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        import os
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                print(f"{Fore.RED}Error creating database directory: {str(e)}")

    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            print(f"{Fore.GREEN}Database connected: {self.db_path}")
        except Exception as e:
            print(f"{Fore.RED}Error connecting to database: {str(e)}")

    def _create_tables(self):
        """Create database tables if they don't exist"""
        if not self.conn or not self.cursor:
            print(f"{Fore.YELLOW}Database not connected - skipping table creation")
            return

        try:
            # Market data table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    banknifty_spot REAL,
                    india_vix REAL,
                    vix_change_pct REAL,
                    hv REAL,
                    hv_daily REAL
                )
            ''')

            # Option chain snapshots table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS option_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    strike INTEGER,
                    ce_ltp REAL,
                    ce_bid REAL,
                    ce_ask REAL,
                    ce_iv REAL,
                    ce_delta REAL,
                    ce_gamma REAL,
                    ce_theta REAL,
                    ce_vega REAL,
                    ce_oi INTEGER,
                    ce_volume INTEGER,
                    pe_ltp REAL,
                    pe_bid REAL,
                    pe_ask REAL,
                    pe_iv REAL,
                    pe_delta REAL,
                    pe_gamma REAL,
                    pe_theta REAL,
                    pe_vega REAL,
                    pe_oi INTEGER,
                    pe_volume INTEGER
                )
            ''')

            # Volatility metrics table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS volatility_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    iv REAL,
                    hv REAL,
                    iv_percentile REAL,
                    hv_window INTEGER
                )
            ''')

            # PCR metrics table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS pcr_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    pcr_oi REAL,
                    pcr_volume REAL,
                    total_ce_oi INTEGER,
                    total_pe_oi INTEGER,
                    total_ce_volume INTEGER,
                    total_pe_volume INTEGER
                )
            ''')

            # Greeks aggregated data
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS greeks_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    atm_strike INTEGER,
                    ce_delta REAL,
                    ce_gamma REAL,
                    ce_theta REAL,
                    ce_vega REAL,
                    pe_delta REAL,
                    pe_gamma REAL,
                    pe_theta REAL,
                    pe_vega REAL
                )
            ''')

            self.conn.commit()
            print(f"{Fore.GREEN}Database tables created successfully")

        except Exception as e:
            print(f"{Fore.RED}Error creating tables: {str(e)}")

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
                    vix_change_pct, hv, hv_daily
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                market_data.get('timestamp', datetime.now(self.ist)),
                market_data.get('spot'),
                market_data.get('vix'),
                market_data.get('vix_change_pct'),
                market_data.get('hv'),
                market_data.get('hv_daily')
            ))

            self.conn.commit()

        except Exception as e:
            print(f"{Fore.RED}Error logging market data: {str(e)}")

    def log_option_chain(self, chain_df: pd.DataFrame):
        """
        Log option chain snapshot

        Args:
            chain_df: Option chain DataFrame
        """
        try:
            timestamp = datetime.now(self.ist)

            for _, row in chain_df.iterrows():
                self.cursor.execute('''
                    INSERT INTO option_chain (
                        timestamp, strike,
                        ce_ltp, ce_bid, ce_ask, ce_iv, ce_delta, ce_gamma, ce_theta, ce_vega, ce_oi, ce_volume,
                        pe_ltp, pe_bid, pe_ask, pe_iv, pe_delta, pe_gamma, pe_theta, pe_vega, pe_oi, pe_volume
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    row['strike'],
                    row['ce_ltp'], row['ce_bid'], row['ce_ask'], row['ce_iv'],
                    row['ce_delta'], row['ce_gamma'], row['ce_theta'], row['ce_vega'],
                    row['ce_oi'], row['ce_volume'],
                    row['pe_ltp'], row['pe_bid'], row['pe_ask'], row['pe_iv'],
                    row['pe_delta'], row['pe_gamma'], row['pe_theta'], row['pe_vega'],
                    row['pe_oi'], row['pe_volume']
                ))

            self.conn.commit()

        except Exception as e:
            print(f"{Fore.RED}Error logging option chain: {str(e)}")

    def log_volatility(self, volatility_data: Dict):
        """
        Log volatility metrics

        Args:
            volatility_data: Volatility data dictionary
        """
        try:
            self.cursor.execute('''
                INSERT INTO volatility_data (
                    timestamp, iv, hv, iv_percentile, hv_window
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                volatility_data.get('timestamp', datetime.now(self.ist)),
                volatility_data.get('iv'),
                volatility_data.get('hv'),
                volatility_data.get('iv_percentile'),
                volatility_data.get('hv_window')
            ))

            self.conn.commit()

        except Exception as e:
            print(f"{Fore.RED}Error logging volatility data: {str(e)}")

    def log_pcr(self, pcr_data: Dict):
        """
        Log PCR metrics

        Args:
            pcr_data: PCR data dictionary
        """
        try:
            self.cursor.execute('''
                INSERT INTO pcr_data (
                    timestamp, pcr_oi, pcr_volume,
                    total_ce_oi, total_pe_oi, total_ce_volume, total_pe_volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                pcr_data.get('timestamp', datetime.now(self.ist)),
                pcr_data.get('pcr_oi'),
                pcr_data.get('pcr_volume'),
                pcr_data.get('ce_oi'),
                pcr_data.get('pe_oi'),
                pcr_data.get('ce_volume'),
                pcr_data.get('pe_volume')
            ))

            self.conn.commit()

        except Exception as e:
            print(f"{Fore.RED}Error logging PCR data: {str(e)}")

    def get_market_data(self, days: int = 1) -> pd.DataFrame:
        """
        Get market data for specified days

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with market data
        """
        try:
            cutoff_date = datetime.now(self.ist) - timedelta(days=days)

            query = '''
                SELECT * FROM market_data
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''

            df = pd.read_sql_query(query, self.conn, params=(cutoff_date,))
            return df

        except Exception as e:
            print(f"{Fore.RED}Error fetching market data: {str(e)}")
            return pd.DataFrame()

    def get_option_chain_snapshots(self, hours: int = 1) -> pd.DataFrame:
        """
        Get option chain snapshots for specified hours

        Args:
            hours: Number of hours to look back

        Returns:
            DataFrame with option chain data
        """
        try:
            cutoff_time = datetime.now(self.ist) - timedelta(hours=hours)

            query = '''
                SELECT * FROM option_chain
                WHERE timestamp >= ?
                ORDER BY timestamp DESC, strike ASC
            '''

            df = pd.read_sql_query(query, self.conn, params=(cutoff_time,))
            return df

        except Exception as e:
            print(f"{Fore.RED}Error fetching option chain snapshots: {str(e)}")
            return pd.DataFrame()

    def get_collection_stats(self) -> Dict:
        """
        Get data collection statistics

        Returns:
            Statistics dictionary
        """
        try:
            stats = {}

            # Count market data records
            self.cursor.execute('SELECT COUNT(*) FROM market_data')
            stats['market_data_count'] = self.cursor.fetchone()[0]

            # Count option chain records
            self.cursor.execute('SELECT COUNT(DISTINCT timestamp) FROM option_chain')
            stats['option_chain_count'] = self.cursor.fetchone()[0]

            # Count volatility records
            self.cursor.execute('SELECT COUNT(*) FROM volatility_data')
            stats['volatility_count'] = self.cursor.fetchone()[0]

            # Get date range
            self.cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM market_data')
            min_date, max_date = self.cursor.fetchone()
            stats['data_start'] = min_date
            stats['data_end'] = max_date

            return stats

        except Exception as e:
            print(f"{Fore.RED}Error fetching collection stats: {str(e)}")
            return {}

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print(f"{Fore.YELLOW}Database connection closed")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


if __name__ == "__main__":
    print("Testing Database...")
    db = MarketDataDatabase()

    # Test market data logging
    sample_market_data = {
        'timestamp': datetime.now(),
        'spot': 48250.50,
        'vix': 15.75,
        'vix_change_pct': -1.2,
        'hv': 18.5,
        'hv_daily': 1.2
    }

    db.log_market_data(sample_market_data)
    print(f"Logged market data sample")

    # Test stats
    stats = db.get_collection_stats()
    print(f"\nCollection Stats: {stats}")

    print("\nDatabase module working successfully!")
