"""
Data Analysis Module
Analyze option chain data, volatility, and Greeks
No trading signals - data analysis only
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz
from colorama import Fore

from config import INSTRUMENT_CONFIG, MARKET_DATA, ANALYSIS_CONFIG


class OptionsDataAnalyzer:
    """Analyze BankNifty options data"""

    def __init__(self, kite, market_monitor):
        """
        Initialize the data analyzer

        Args:
            kite: KiteConnect instance
            market_monitor: MarketMonitor instance
        """
        self.kite = kite
        self.market_monitor = market_monitor
        self.ist = pytz.timezone('Asia/Kolkata')

        # Historical data
        self.iv_history = []
        self.spot_history = []

    def get_banknifty_spot(self) -> float:
        """
        Get current BankNifty spot price

        Returns:
            Current spot price
        """
        try:
            if self.kite:
                quote = self.kite.quote(INSTRUMENT_CONFIG['index_symbol'])
                spot = quote[INSTRUMENT_CONFIG['index_symbol']]["last_price"]

                # Store in history
                self.spot_history.append({
                    'timestamp': datetime.now(self.ist),
                    'price': spot
                })

                # Keep only last 100 entries
                if len(self.spot_history) > 100:
                    self.spot_history = self.spot_history[-100:]

                return spot
            else:
                # Mock data for testing
                return 48000.0
        except Exception as e:
            print(f"{Fore.RED}Error fetching spot price: {str(e)}")
            return 0.0

    def get_option_chain(self) -> pd.DataFrame:
        """
        Fetch BankNifty option chain data

        Returns:
            DataFrame with option chain data
        """
        try:
            spot = self.get_banknifty_spot()
            if spot == 0:
                return pd.DataFrame()

            # Get next weekly expiry
            expiry = self._get_next_expiry()

            # Calculate strike range from config
            strike_range = MARKET_DATA['strike_range']
            lower_strike = int((spot - strike_range) / 100) * 100
            upper_strike = int((spot + strike_range) / 100) * 100

            # Generate strike list
            strikes = list(range(lower_strike, upper_strike + 100, 100))

            # Fetch option chain from Kite
            if self.kite:
                instruments = []
                for strike in strikes:
                    # CE
                    ce_symbol = f"BANKNIFTY{expiry}{strike}CE"
                    instruments.append(f"NFO:{ce_symbol}")
                    # PE
                    pe_symbol = f"BANKNIFTY{expiry}{strike}PE"
                    instruments.append(f"NFO:{pe_symbol}")

                # Fetch quotes
                quotes = self.kite.quote(instruments)

                # Parse into DataFrame
                chain_data = []
                for strike in strikes:
                    ce_symbol = f"NFO:BANKNIFTY{expiry}{strike}CE"
                    pe_symbol = f"NFO:BANKNIFTY{expiry}{strike}PE"

                    ce_data = quotes.get(ce_symbol, {})
                    pe_data = quotes.get(pe_symbol, {})

                    chain_data.append({
                        'strike': strike,
                        'ce_ltp': ce_data.get('last_price', 0),
                        'ce_bid': ce_data.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                        'ce_ask': ce_data.get('depth', {}).get('sell', [{}])[0].get('price', 0),
                        'ce_iv': ce_data.get('iv', 0),
                        'ce_delta': ce_data.get('delta', 0),
                        'ce_gamma': ce_data.get('gamma', 0),
                        'ce_theta': ce_data.get('theta', 0),
                        'ce_vega': ce_data.get('vega', 0),
                        'ce_oi': ce_data.get('oi', 0),
                        'ce_volume': ce_data.get('volume', 0),
                        'pe_ltp': pe_data.get('last_price', 0),
                        'pe_bid': pe_data.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                        'pe_ask': pe_data.get('depth', {}).get('sell', [{}])[0].get('price', 0),
                        'pe_iv': pe_data.get('iv', 0),
                        'pe_delta': pe_data.get('delta', 0),
                        'pe_gamma': pe_data.get('gamma', 0),
                        'pe_theta': pe_data.get('theta', 0),
                        'pe_vega': pe_data.get('vega', 0),
                        'pe_oi': pe_data.get('oi', 0),
                        'pe_volume': pe_data.get('volume', 0),
                    })

                df = pd.DataFrame(chain_data)

            else:
                # Mock option chain for testing
                df = self._generate_mock_option_chain(spot, strikes)

            # Add ATM flag
            df['atm_distance'] = abs(df['strike'] - spot)
            df['is_atm'] = df['atm_distance'] == df['atm_distance'].min()

            # Add timestamp
            df['timestamp'] = datetime.now(self.ist)

            return df

        except Exception as e:
            print(f"{Fore.RED}Error fetching option chain: {str(e)}")
            return pd.DataFrame()

    def _generate_mock_option_chain(self, spot: float, strikes: List[int]) -> pd.DataFrame:
        """Generate mock option chain for testing"""
        chain_data = []

        for strike in strikes:
            # Simple Black-Scholes approximation for mock data
            distance = abs(strike - spot)
            ce_premium = max(spot - strike, 0) + distance * 0.1
            pe_premium = max(strike - spot, 0) + distance * 0.1

            # Mock greeks
            ce_delta = 0.5 if strike == int(spot / 100) * 100 else (0.7 if strike < spot else 0.3)
            pe_delta = -0.5 if strike == int(spot / 100) * 100 else (-0.3 if strike < spot else -0.7)

            chain_data.append({
                'strike': strike,
                'ce_ltp': ce_premium,
                'ce_bid': ce_premium - 1,
                'ce_ask': ce_premium + 1,
                'ce_iv': 18 + np.random.uniform(-2, 2),
                'ce_delta': ce_delta,
                'ce_gamma': 0.002,
                'ce_theta': -50,
                'ce_vega': 30,
                'ce_oi': np.random.randint(1000, 50000),
                'ce_volume': np.random.randint(100, 5000),
                'pe_ltp': pe_premium,
                'pe_bid': pe_premium - 1,
                'pe_ask': pe_premium + 1,
                'pe_iv': 18 + np.random.uniform(-2, 2),
                'pe_delta': pe_delta,
                'pe_gamma': 0.002,
                'pe_theta': -50,
                'pe_vega': 30,
                'pe_oi': np.random.randint(1000, 50000),
                'pe_volume': np.random.randint(100, 5000),
            })

        return pd.DataFrame(chain_data)

    def _get_next_expiry(self) -> str:
        """
        Get next weekly expiry date for BankNifty

        Returns:
            Expiry date string in format YYMMDD
        """
        today = datetime.now(self.ist)
        days_until_wednesday = (2 - today.weekday()) % 7  # Wednesday = 2

        if days_until_wednesday == 0 and today.hour >= 15:
            # If it's Wednesday after 3:30 PM, get next Wednesday
            days_until_wednesday = 7

        expiry_date = today + timedelta(days=days_until_wednesday)
        return expiry_date.strftime('%y%m%d')

    def calculate_iv_percentile(self) -> float:
        """
        Calculate IV percentile based on historical data

        Returns:
            IV percentile (0-100)
        """
        try:
            chain = self.get_option_chain()
            if chain.empty:
                return 50.0

            # Get ATM IV
            atm_row = chain[chain['is_atm']].iloc[0]
            current_iv = (atm_row['ce_iv'] + atm_row['pe_iv']) / 2

            # Store in history
            self.iv_history.append({
                'timestamp': datetime.now(self.ist),
                'iv': current_iv
            })

            # Keep only last 60 days worth of data
            window = ANALYSIS_CONFIG['iv_percentile_window']
            cutoff_date = datetime.now(self.ist) - timedelta(days=window)
            self.iv_history = [x for x in self.iv_history if x['timestamp'] > cutoff_date]

            if len(self.iv_history) < 10:
                return 50.0  # Not enough data

            # Calculate percentile
            iv_values = [x['iv'] for x in self.iv_history]
            percentile = (sum(1 for x in iv_values if x < current_iv) / len(iv_values)) * 100

            return percentile

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not calculate IV percentile: {str(e)}")
            return 50.0

    def calculate_pcr(self, chain: pd.DataFrame = None) -> Dict:
        """
        Calculate Put-Call Ratio

        Args:
            chain: Option chain DataFrame (optional, will fetch if not provided)

        Returns:
            PCR metrics dictionary
        """
        try:
            if chain is None:
                chain = self.get_option_chain()

            if chain.empty:
                return {'pcr_oi': 0, 'pcr_volume': 0}

            # OI-based PCR
            total_ce_oi = chain['ce_oi'].sum()
            total_pe_oi = chain['pe_oi'].sum()
            pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

            # Volume-based PCR
            total_ce_volume = chain['ce_volume'].sum()
            total_pe_volume = chain['pe_volume'].sum()
            pcr_volume = total_pe_volume / total_ce_volume if total_ce_volume > 0 else 0

            return {
                'pcr_oi': pcr_oi,
                'pcr_volume': pcr_volume,
                'ce_oi': total_ce_oi,
                'pe_oi': total_pe_oi,
                'ce_volume': total_ce_volume,
                'pe_volume': total_pe_volume,
                'timestamp': datetime.now(self.ist)
            }

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not calculate PCR: {str(e)}")
            return {'pcr_oi': 0, 'pcr_volume': 0}

    def analyze_max_pain(self, chain: pd.DataFrame = None) -> float:
        """
        Calculate Max Pain strike price

        Args:
            chain: Option chain DataFrame

        Returns:
            Max pain strike price
        """
        try:
            if chain is None:
                chain = self.get_option_chain()

            if chain.empty:
                return 0

            max_pain_strike = 0
            min_pain_value = float('inf')

            for strike in chain['strike'].values:
                # Calculate total pain at this strike
                ce_pain = chain[chain['strike'] >= strike]['ce_oi'].sum() * INSTRUMENT_CONFIG['lot_size']
                pe_pain = chain[chain['strike'] <= strike]['pe_oi'].sum() * INSTRUMENT_CONFIG['lot_size']

                total_pain = ce_pain + pe_pain

                if total_pain < min_pain_value:
                    min_pain_value = total_pain
                    max_pain_strike = strike

            return max_pain_strike

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not calculate max pain: {str(e)}")
            return 0


if __name__ == "__main__":
    print("Testing Data Analyzer...")
    print("Data analyzer module loaded successfully")
