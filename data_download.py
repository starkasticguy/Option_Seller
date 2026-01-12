"""
Zerodha Option Chain Data Download Module
Downloads NIFTY option chain data
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from auth import ZerodhaAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionChainDownloader:
    def __init__(self, kite):
        """
        Initialize option chain downloader

        Args:
            kite: Authenticated KiteConnect instance
        """
        self.kite = kite

    def get_nifty_spot_price(self):
        """Get current NIFTY spot price"""
        try:
            quote = self.kite.quote("NSE:NIFTY 50")
            spot_price = quote["NSE:NIFTY 50"]["last_price"]
            logger.info(f"NIFTY Spot Price: {spot_price}")
            return spot_price
        except Exception as e:
            logger.error(f"Error fetching NIFTY spot price: {e}")
            raise

    def get_option_instruments(self, expiry_date=None):
        """
        Get NIFTY option instruments

        Args:
            expiry_date: Specific expiry date (YYYY-MM-DD) or None for nearest expiry

        Returns:
            DataFrame: Option instruments
        """
        try:
            instruments = self.kite.instruments("NFO")
            df = pd.DataFrame(instruments)

            # Filter for NIFTY options
            nifty_options = df[
                (df['name'] == 'NIFTY') &
                (df['instrument_type'].isin(['CE', 'PE']))
                ].copy()

            if expiry_date:
                nifty_options = nifty_options[
                    nifty_options['expiry'] == pd.to_datetime(expiry_date)
                    ]
            else:
                # Get nearest expiry
                nifty_options['expiry'] = pd.to_datetime(nifty_options['expiry'])
                nearest_expiry = nifty_options['expiry'].min()
                nifty_options = nifty_options[nifty_options['expiry'] == nearest_expiry]
                logger.info(f"Using nearest expiry: {nearest_expiry.date()}")

            logger.info(f"Found {len(nifty_options)} NIFTY option contracts")
            return nifty_options
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            raise

    def get_option_chain_data(self, expiry_date=None, strike_range=500):
        """
        Download complete option chain data

        Args:
            expiry_date: Specific expiry date or None for nearest
            strike_range: Strike range around spot price (e.g., +/- 500 points)

        Returns:
            DataFrame: Option chain with quotes
        """
        try:
            # Get spot price
            spot_price = self.get_nifty_spot_price()

            # Get option instruments
            instruments = self.get_option_instruments(expiry_date)

            # Filter strikes around spot price
            if strike_range:
                instruments = instruments[
                    (instruments['strike'] >= spot_price - strike_range) &
                    (instruments['strike'] <= spot_price + strike_range)
                    ]

            # Get quotes for all instruments
            symbols = instruments['tradingsymbol'].tolist()
            exchange_symbols = [f"NFO:{symbol}" for symbol in symbols]

            logger.info(f"Fetching quotes for {len(exchange_symbols)} option contracts...")
            quotes = self.kite.quote(exchange_symbols)

            # Prepare data for DataFrame
            option_data = []
            for symbol in exchange_symbols:
                if symbol in quotes:
                    q = quotes[symbol]
                    instrument_row = instruments[
                        instruments['tradingsymbol'] == symbol.split(':')[1]
                        ].iloc[0]

                    option_data.append({
                        'timestamp': datetime.now(),
                        'symbol': instrument_row['tradingsymbol'],
                        'expiry': instrument_row['expiry'],
                        'strike': instrument_row['strike'],
                        'option_type': instrument_row['instrument_type'],
                        'spot_price': spot_price,
                        'last_price': q.get('last_price', 0),
                        'volume': q.get('volume', 0),
                        'buy_qty': q.get('depth', {}).get('buy', [{}])[0].get('quantity', 0) if q.get('depth') else 0,
                        'sell_qty': q.get('depth', {}).get('sell', [{}])[0].get('quantity', 0) if q.get('depth') else 0,
                        'oi': q.get('oi', 0),
                        'oi_day_high': q.get('oi_day_high', 0),
                        'oi_day_low': q.get('oi_day_low', 0),
                        'bid_price': q.get('depth', {}).get('buy', [{}])[0].get('price', 0) if q.get('depth') else 0,
                        'ask_price': q.get('depth', {}).get('sell', [{}])[0].get('price', 0) if q.get('depth') else 0,
                        'change': q.get('net_change', 0),
                        'change_percent': q.get('change', 0)
                    })

            df = pd.DataFrame(option_data)
            logger.info(f"Successfully downloaded data for {len(df)} options")
            return df

        except Exception as e:
            logger.error(f"Error downloading option chain: {e}")
            raise

    def format_option_chain(self, df):
        """
        Format option chain in traditional format (CE and PE side by side)

        Args:
            df: Raw option chain DataFrame

        Returns:
            DataFrame: Formatted option chain
        """
        ce_data = df[df['option_type'] == 'CE'].sort_values('strike')
        pe_data = df[df['option_type'] == 'PE'].sort_values('strike')

        ce_data = ce_data.add_suffix('_CE')
        pe_data = pe_data.add_suffix('_PE')

        # Merge on strike price
        formatted = pd.merge(
            ce_data,
            pe_data,
            left_on='strike_CE',
            right_on='strike_PE',
            how='outer'
        )

        return formatted


def main():
    """Example usage"""
    # Load credentials
    API_KEY = "pi2xk2iokf0u2nt3"
    API_SECRET = "ssifv93g7g6xo9lofpduwwzy4n45q3d2"

    # Authenticate
    auth = ZerodhaAuth(API_KEY, API_SECRET)
    auth.load_token()
    kite = auth.get_kite_instance()

    # Download option chain
    downloader = OptionChainDownloader(kite)

    # Get option chain for nearest expiry
    option_chain = downloader.get_option_chain_data(strike_range=2000)

    # Display summary
    print("\n=== Option Chain Summary ===")
    print(f"Total contracts: {len(option_chain)}")
    print(f"Timestamp: {option_chain['timestamp'].iloc[0]}")
    print(f"Spot Price: {option_chain['spot_price'].iloc[0]}")
    print(f"Expiry: {option_chain['expiry'].iloc[0].date()}")

    # Display sample data
    print("\n=== Sample Data (First 5 Calls) ===")
    calls = option_chain[option_chain['option_type'] == 'CE'].head()
    print(calls[['strike', 'last_price', 'oi', 'volume']])

    # Save to CSV for testing
    option_chain.to_csv('option_chain_data.csv', index=False)
    print("\nData saved to option_chain_data.csv")


if __name__ == "__main__":
    main()