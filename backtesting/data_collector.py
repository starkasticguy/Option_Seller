"""
Historical Data Collector
Downloads and processes historical market data from Zerodha
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path
import pytz

from backtest_config import DATA_CONFIG, EVENTS_CONFIG, OPTION_PRICING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """
    Collects historical data for backtesting
    """

    def __init__(self, kite=None):
        """
        Initialize data collector

        Args:
            kite: KiteConnect instance (optional, for live data download)
        """
        self.kite = kite
        self.config = DATA_CONFIG
        self.ist = pytz.timezone('Asia/Kolkata')

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config['data_dir'],
            self.config['historical_data_dir'],
            self.config['backtest_results_dir'],
            self.config['cache_dir'],
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def download_banknifty_historical(
        self,
        start_date: str,
        end_date: str,
        interval: str = '5minute'
    ) -> pd.DataFrame:
        """
        Download BankNifty historical data from Zerodha

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (5minute, 15minute, day, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Downloading BankNifty data from {start_date} to {end_date}")

        if self.kite is None:
            logger.warning("No KiteConnect instance provided, using mock data")
            return self._generate_mock_banknifty_data(start_date, end_date, interval)

        try:
            # Convert interval to Zerodha format
            interval_map = {
                '1minute': 'minute',
                '5minute': '5minute',
                '15minute': '15minute',
                '30minute': '30minute',
                '60minute': '60minute',
                'day': 'day'
            }
            kite_interval = interval_map.get(interval, '5minute')

            # Zerodha API has a limit on historical data (60 days for intraday)
            # Need to download in chunks
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            all_data = []

            # For intraday data, download in 60-day chunks
            if interval != 'day':
                chunk_days = 60
            else:
                chunk_days = 365

            current_start = start_dt
            while current_start < end_dt:
                current_end = min(current_start + timedelta(days=chunk_days), end_dt)

                logger.info(f"  Downloading chunk: {current_start.date()} to {current_end.date()}")

                # Download data
                instrument_token = self._get_instrument_token('NSE:NIFTY BANK')
                data = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=current_start,
                    to_date=current_end,
                    interval=kite_interval
                )

                if data:
                    all_data.extend(data)

                current_start = current_end + timedelta(days=1)

            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            logger.info(f"Downloaded {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            logger.warning("Falling back to mock data")
            return self._generate_mock_banknifty_data(start_date, end_date, interval)

    def download_vix_historical(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download India VIX historical data

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with VIX data
        """
        logger.info(f"Downloading India VIX data from {start_date} to {end_date}")

        if self.kite is None:
            logger.warning("No KiteConnect instance provided, using mock data")
            return self._generate_mock_vix_data(start_date, end_date)

        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # VIX is daily data
            instrument_token = self._get_instrument_token('INDIA VIX')
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_dt,
                to_date=end_dt,
                interval='day'
            )

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.rename(columns={'close': 'VIX'})

            # Keep only VIX column
            df = df[['VIX']]

            logger.info(f"Downloaded {len(df)} VIX records")
            return df

        except Exception as e:
            logger.error(f"Error downloading VIX data: {str(e)}")
            logger.warning("Falling back to mock VIX data")
            return self._generate_mock_vix_data(start_date, end_date)

    def _get_instrument_token(self, symbol: str) -> int:
        """Get instrument token for symbol"""
        # This is a placeholder - in real implementation, you would
        # query the instruments list from Zerodha
        token_map = {
            'NSE:NIFTY BANK': 260105,
            'INDIA VIX': 264969,
        }
        return token_map.get(symbol, 0)

    def _generate_mock_banknifty_data(
        self,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Generate mock BankNifty data for testing

        Args:
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with mock OHLCV data
        """
        logger.info("Generating mock BankNifty data...")

        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate timestamps based on interval
        if interval == '5minute':
            freq = '5T'
        elif interval == '15minute':
            freq = '15T'
        elif interval == 'day':
            freq = 'D'
        else:
            freq = '5T'

        # Generate date range (only trading hours)
        dates = pd.date_range(start=start_dt, end=end_dt, freq=freq)

        # Filter for market hours (9:15 AM to 3:30 PM IST)
        if interval != 'day':
            market_open = time(9, 15)
            market_close = time(15, 30)
            dates = [
                d for d in dates
                if d.time() >= market_open and d.time() <= market_close
                and d.weekday() < 5  # Monday to Friday
            ]

        # Remove market holidays
        dates = [d for d in dates if not self._is_market_holiday(d)]

        # Generate realistic price data using GBM
        n_points = len(dates)
        initial_price = 45000  # Starting BankNifty price

        # Simulate price movement
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.015, n_points)  # Drift and volatility

        # Add some trends and regimes
        # COVID crash (March 2020)
        covid_crash_start = pd.to_datetime('2020-03-01')
        covid_crash_end = pd.to_datetime('2020-04-01')

        # Bull run (2021)
        bull_run_start = pd.to_datetime('2021-01-01')
        bull_run_end = pd.to_datetime('2021-12-31')

        # Modify returns based on regime
        for i, d in enumerate(dates):
            if covid_crash_start <= d <= covid_crash_end:
                returns[i] = np.random.normal(-0.002, 0.04, 1)[0]  # High volatility, negative drift
            elif bull_run_start <= d <= bull_run_end:
                returns[i] = np.random.normal(0.0005, 0.012, 1)[0]  # Positive drift

        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLC from prices
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC
            daily_vol = abs(np.random.normal(0, 0.01, 1)[0])
            high = price * (1 + daily_vol)
            low = price * (1 - daily_vol)
            open_ = price * (1 + np.random.normal(0, daily_vol/2, 1)[0])
            close = price

            volume = np.random.randint(100000, 1000000)

            data.append({
                'Open': open_,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })

        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'

        logger.info(f"Generated {len(df)} mock records")
        return df

    def _generate_mock_vix_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate mock VIX data"""
        logger.info("Generating mock VIX data...")

        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate date range (daily)
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')

        # Filter for trading days
        dates = [d for d in dates if d.weekday() < 5 and not self._is_market_holiday(d)]

        # Generate VIX values
        np.random.seed(43)
        n_points = len(dates)

        # Base VIX around 15-20
        vix_values = np.random.normal(17, 3, n_points)

        # Add regime changes
        covid_crash_start = pd.to_datetime('2020-03-01')
        covid_crash_end = pd.to_datetime('2020-04-30')

        for i, d in enumerate(dates):
            if covid_crash_start <= d <= covid_crash_end:
                vix_values[i] = np.random.normal(35, 10, 1)[0]  # Very high VIX

        # Ensure VIX is in reasonable range
        vix_values = np.clip(vix_values, 10, 80)

        df = pd.DataFrame({'VIX': vix_values}, index=dates)
        df.index.name = 'date'

        logger.info(f"Generated {len(df)} mock VIX records")
        return df

    def _is_market_holiday(self, date: pd.Timestamp) -> bool:
        """Check if date is a market holiday"""
        date_str = date.strftime('%Y-%m-%d')
        return date_str in EVENTS_CONFIG['market_holidays']

    def calculate_realized_volatility(
        self,
        prices: pd.Series,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate realized volatility for multiple windows

        Args:
            prices: Series of prices
            windows: List of window sizes (default from config)

        Returns:
            DataFrame with realized volatility for each window
        """
        if windows is None:
            windows = OPTION_PRICING_CONFIG['realized_vol_windows']

        logger.info(f"Calculating realized volatility for windows: {windows}")

        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))

        result = pd.DataFrame(index=prices.index)

        for window in windows:
            # Rolling standard deviation
            rolling_std = log_returns.rolling(window=window).std()

            # Annualize (assuming 252 trading days per year, but using 365 for intraday)
            annualized_vol = rolling_std * np.sqrt(252)

            result[f'RV_{window}d'] = annualized_vol

        return result

    def combine_and_process_data(
        self,
        spot_data: pd.DataFrame,
        vix_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine spot and VIX data, calculate additional features

        Args:
            spot_data: BankNifty spot data
            vix_data: India VIX data

        Returns:
            Combined DataFrame with all features
        """
        logger.info("Combining and processing data...")

        # If spot_data has intraday data, merge VIX (daily) forward-fill
        combined = spot_data.copy()

        # Merge VIX data
        if isinstance(spot_data.index, pd.DatetimeIndex):
            # Convert VIX index to date only for merging
            vix_daily = vix_data.copy()
            vix_daily.index = vix_daily.index.date

            # Add date column to spot data
            combined['date_only'] = combined.index.date

            # Merge
            combined = combined.merge(
                vix_daily,
                left_on='date_only',
                right_index=True,
                how='left'
            )

            # Forward fill VIX for intraday data
            combined['VIX'] = combined['VIX'].fillna(method='ffill')

            # Drop helper column
            combined = combined.drop('date_only', axis=1)

        # Calculate realized volatility
        rv_data = self.calculate_realized_volatility(combined['Close'])
        combined = pd.concat([combined, rv_data], axis=1)

        # Calculate returns
        combined['Returns'] = combined['Close'].pct_change()
        combined['Log_Returns'] = np.log(combined['Close'] / combined['Close'].shift(1))

        # Calculate price statistics
        combined['High_Low_Range'] = combined['High'] - combined['Low']
        combined['Range_Pct'] = combined['High_Low_Range'] / combined['Close'] * 100

        # Add day of week
        combined['DayOfWeek'] = combined.index.dayofweek

        # Add time features for intraday
        if len(combined) > 0 and isinstance(combined.index[0], pd.Timestamp):
            if combined.index[0].time() != time(0, 0):
                combined['Hour'] = combined.index.hour
                combined['Minute'] = combined.index.minute

        # Fill missing data if configured
        if DATA_CONFIG['fill_missing_data']:
            combined = combined.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"Combined data shape: {combined.shape}")
        return combined

    def save_to_parquet(
        self,
        data: pd.DataFrame,
        filename: str
    ) -> None:
        """
        Save DataFrame to parquet format

        Args:
            data: DataFrame to save
            filename: Output filename
        """
        filepath = os.path.join(self.config['historical_data_dir'], filename)
        data.to_parquet(filepath, compression='gzip')
        logger.info(f"Saved data to {filepath}")

    def load_from_parquet(
        self,
        filename: str
    ) -> pd.DataFrame:
        """
        Load DataFrame from parquet format

        Args:
            filename: Input filename

        Returns:
            Loaded DataFrame
        """
        filepath = os.path.join(self.config['historical_data_dir'], filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_parquet(filepath)
        logger.info(f"Loaded data from {filepath} ({len(df)} records)")
        return df

    def collect_and_save_data(
        self,
        start_date: str = None,
        end_date: str = None,
        interval: str = '5minute'
    ) -> pd.DataFrame:
        """
        Main method to collect and save all required data

        Args:
            start_date: Start date (default from config)
            end_date: End date (default from config)
            interval: Data interval

        Returns:
            Combined DataFrame
        """
        # Use config dates if not provided
        start_date = start_date or DATA_CONFIG['start_date']
        end_date = end_date or DATA_CONFIG['end_date']

        logger.info("=" * 60)
        logger.info("Starting data collection...")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("=" * 60)

        # Download BankNifty data
        spot_data = self.download_banknifty_historical(start_date, end_date, interval)
        self.save_to_parquet(spot_data, 'banknifty_spot.parquet')

        # Download VIX data
        vix_data = self.download_vix_historical(start_date, end_date)
        self.save_to_parquet(vix_data, 'india_vix.parquet')

        # Combine and process
        combined_data = self.combine_and_process_data(spot_data, vix_data)
        self.save_to_parquet(combined_data, 'combined_market_data.parquet')

        logger.info("=" * 60)
        logger.info("✅ Data collection completed!")
        logger.info(f"Total records: {len(combined_data)}")
        logger.info(f"Date range: {combined_data.index[0]} to {combined_data.index[-1]}")
        logger.info(f"Columns: {list(combined_data.columns)}")
        logger.info("=" * 60)

        return combined_data

    def get_expiry_dates(
        self,
        start_date: str,
        end_date: str
    ) -> List[pd.Timestamp]:
        """
        Get all weekly expiry dates (Wednesdays) in the date range

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of expiry dates
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate all Wednesdays
        dates = pd.date_range(start=start_dt, end=end_dt, freq='W-WED')

        # Remove holidays
        expiry_dates = [d for d in dates if not self._is_market_holiday(d)]

        # If Wednesday is holiday, expiry moves to previous trading day
        adjusted_expiry_dates = []
        for exp_date in expiry_dates:
            current_date = exp_date
            while self._is_market_holiday(current_date):
                current_date -= timedelta(days=1)
            adjusted_expiry_dates.append(current_date)

        logger.info(f"Found {len(adjusted_expiry_dates)} expiry dates")
        return adjusted_expiry_dates


def main():
    """Main function to collect data"""
    print("\n" + "=" * 60)
    print("  BankNifty Historical Data Collector")
    print("=" * 60 + "\n")

    # Initialize collector (no Kite instance - will use mock data)
    collector = HistoricalDataCollector(kite=None)

    # Collect data
    try:
        combined_data = collector.collect_and_save_data()

        # Show sample
        print("\nSample data:")
        print(combined_data.head(10))

        print("\nData statistics:")
        print(combined_data.describe())

    except Exception as e:
        logger.error(f"Error collecting data: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
