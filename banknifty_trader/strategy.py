"""
Trading Strategy Module
Contains core trading logic for BankNifty options
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pytz
from colorama import Fore, Style
import math

from config import TRADING_PARAMS, INSTRUMENT_CONFIG, MARKET_DATA


class BankNiftyOptionsTrader:
    """Main trading strategy class for BankNifty options"""

    def __init__(self, kite, market_monitor):
        """
        Initialize the trading strategy

        Args:
            kite: KiteConnect instance
            market_monitor: MarketMonitor instance
        """
        self.kite = kite
        self.market_monitor = market_monitor
        self.ist = pytz.timezone('Asia/Kolkata')

        # Position tracking
        self.current_position = None
        self.entry_time = None
        self.entry_premium = {}
        self.entry_strikes = {}
        self.position_greeks = {}

        # Historical data
        self.iv_history = []
        self.signal_history = []

    def get_banknifty_spot(self) -> float:
        """
        Get current BankNifty spot price

        Returns:
            Current spot price
        """
        try:
            if self.kite:
                quote = self.kite.quote("NSE:NIFTY BANK")
                spot = quote["NSE:NIFTY BANK"]["last_price"]
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

            # Calculate strike range (±1000 points from spot)
            strike_range = 1000
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
                        'pe_ltp': pe_data.get('last_price', 0),
                        'pe_bid': pe_data.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                        'pe_ask': pe_data.get('depth', {}).get('sell', [{}])[0].get('price', 0),
                        'pe_iv': pe_data.get('iv', 0),
                        'pe_delta': pe_data.get('delta', 0),
                        'pe_gamma': pe_data.get('gamma', 0),
                        'pe_theta': pe_data.get('theta', 0),
                        'pe_vega': pe_data.get('vega', 0),
                        'pe_oi': pe_data.get('oi', 0),
                    })

                df = pd.DataFrame(chain_data)

            else:
                # Mock option chain for testing
                df = self._generate_mock_option_chain(spot, strikes)

            # Add ATM flag
            df['atm_distance'] = abs(df['strike'] - spot)
            df['is_atm'] = df['atm_distance'] == df['atm_distance'].min()

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
                'pe_ltp': pe_premium,
                'pe_bid': pe_premium - 1,
                'pe_ask': pe_premium + 1,
                'pe_iv': 18 + np.random.uniform(-2, 2),
                'pe_delta': pe_delta,
                'pe_gamma': 0.002,
                'pe_theta': -50,
                'pe_vega': 30,
                'pe_oi': np.random.randint(1000, 50000),
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

            # Store in history (keep last 60 days)
            self.iv_history.append({
                'timestamp': datetime.now(self.ist),
                'iv': current_iv
            })

            # Keep only last 60 days
            cutoff_date = datetime.now(self.ist) - timedelta(days=60)
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

    def select_strikes(self, strategy_type: str = None) -> Dict:
        """
        Select optimal strikes based on strategy

        Args:
            strategy_type: Strategy type (default from config)

        Returns:
            Dict with CE and PE strikes
        """
        if strategy_type is None:
            strategy_type = TRADING_PARAMS['strategy_type']

        chain = self.get_option_chain()
        if chain.empty:
            return {}

        spot = self.get_banknifty_spot()
        selection_method = TRADING_PARAMS['strike_selection']

        if selection_method == 'DELTA':
            # Select based on delta targets
            ce_target_delta = abs(TRADING_PARAMS['ce_delta_target'])
            pe_target_delta = abs(TRADING_PARAMS['pe_delta_target'])

            # Find CE strike closest to target delta
            chain['ce_delta_diff'] = abs(abs(chain['ce_delta']) - ce_target_delta)
            ce_strike_row = chain.loc[chain['ce_delta_diff'].idxmin()]
            ce_strike = int(ce_strike_row['strike'])

            # Find PE strike closest to target delta
            chain['pe_delta_diff'] = abs(abs(chain['pe_delta']) - pe_target_delta)
            pe_strike_row = chain.loc[chain['pe_delta_diff'].idxmin()]
            pe_strike = int(pe_strike_row['strike'])

        elif selection_method == 'ATM_OFFSET':
            # Select based on distance from ATM
            offset = TRADING_PARAMS['atm_offset']
            atm_strike = int(spot / 100) * 100

            ce_strike = atm_strike + offset
            pe_strike = atm_strike - offset

        else:  # PREMIUM based
            # Select based on premium range
            min_premium = TRADING_PARAMS['min_premium']
            max_premium = TRADING_PARAMS['max_premium']

            # Filter by premium range
            ce_candidates = chain[
                (chain['ce_ltp'] >= min_premium) &
                (chain['ce_ltp'] <= max_premium)
            ]
            pe_candidates = chain[
                (chain['pe_ltp'] >= min_premium) &
                (chain['pe_ltp'] <= max_premium)
            ]

            if not ce_candidates.empty:
                ce_strike_row = ce_candidates.iloc[len(ce_candidates) // 2]
                ce_strike = int(ce_strike_row['strike'])
            else:
                ce_strike = int(spot / 100) * 100 + 200

            if not pe_candidates.empty:
                pe_strike_row = pe_candidates.iloc[len(pe_candidates) // 2]
                pe_strike = int(pe_strike_row['strike'])
            else:
                pe_strike = int(spot / 100) * 100 - 200

        # Get premium and greek data for selected strikes
        ce_data = chain[chain['strike'] == ce_strike].iloc[0]
        pe_data = chain[chain['strike'] == pe_strike].iloc[0]

        return {
            'CE': {
                'strike': ce_strike,
                'premium': ce_data['ce_ltp'],
                'delta': ce_data['ce_delta'],
                'gamma': ce_data['ce_gamma'],
                'theta': ce_data['ce_theta'],
                'vega': ce_data['ce_vega'],
                'iv': ce_data['ce_iv']
            },
            'PE': {
                'strike': pe_strike,
                'premium': pe_data['pe_ltp'],
                'delta': pe_data['pe_delta'],
                'gamma': pe_data['pe_gamma'],
                'theta': pe_data['pe_theta'],
                'vega': pe_data['pe_vega'],
                'iv': pe_data['pe_iv']
            }
        }

    def calculate_signals(self) -> Dict:
        """
        Generate trading signals based on market conditions

        Returns:
            Signal dictionary with action and parameters
        """
        try:
            # Check if market is open
            market_status = self.market_monitor.get_market_status()
            if not market_status['can_trade']:
                return {
                    'action': 'SKIP',
                    'reason': f"Market not suitable: {', '.join(market_status['reasons'])}",
                    'confidence': 0.0
                }

            # Check VIX levels
            vix_data = market_status['vix']
            vix_low, vix_high = TRADING_PARAMS['vix_range']

            if vix_data['current'] > vix_high:
                return {
                    'action': 'SKIP',
                    'reason': f"VIX too high ({vix_data['current']:.1f} > {vix_high})",
                    'confidence': 0.0
                }

            if vix_data['current'] < vix_low:
                return {
                    'action': 'SKIP',
                    'reason': f"VIX too low ({vix_data['current']:.1f} < {vix_low})",
                    'confidence': 0.0
                }

            # Check for existing position
            if self.current_position:
                return self._generate_position_management_signal()

            # Check IV percentile
            iv_percentile = self.calculate_iv_percentile()
            min_iv_percentile = TRADING_PARAMS.get('iv_percentile_min', 50)

            if iv_percentile < min_iv_percentile:
                return {
                    'action': 'SKIP',
                    'reason': f"IV percentile too low ({iv_percentile:.0f}% < {min_iv_percentile}%)",
                    'confidence': 0.0
                }

            # Select strikes
            strikes = self.select_strikes()
            if not strikes:
                return {
                    'action': 'SKIP',
                    'reason': "Could not select strikes",
                    'confidence': 0.0
                }

            # Calculate confidence based on conditions
            confidence = 0.5  # Base confidence

            # Increase confidence for good VIX
            if 15 <= vix_data['current'] <= 20:
                confidence += 0.2

            # Increase confidence for high IV percentile
            if iv_percentile > 70:
                confidence += 0.15

            # Increase confidence if no events
            if market_status['events']['status'] == 'GREEN':
                confidence += 0.15

            confidence = min(confidence, 1.0)

            # Determine strategy action
            strategy_type = TRADING_PARAMS['strategy_type']

            if strategy_type == 'SHORT_STRANGLE':
                return {
                    'action': 'SELL_STRANGLE',
                    'strikes': strikes,
                    'lots': TRADING_PARAMS['position_size'],
                    'hedge': self._calculate_hedge() if TRADING_PARAMS['use_hedging'] else None,
                    'reason': f"Good conditions: VIX={vix_data['current']:.1f}, IV%ile={iv_percentile:.0f}%",
                    'confidence': confidence,
                    'iv_percentile': iv_percentile,
                    'vix': vix_data['current']
                }

            elif strategy_type == 'LONG_STRANGLE':
                return {
                    'action': 'BUY_STRANGLE',
                    'strikes': strikes,
                    'lots': TRADING_PARAMS['position_size'],
                    'reason': f"Expecting volatility expansion",
                    'confidence': confidence,
                    'iv_percentile': iv_percentile,
                    'vix': vix_data['current']
                }

            else:
                return {
                    'action': 'SKIP',
                    'reason': f"Strategy {strategy_type} not implemented",
                    'confidence': 0.0
                }

        except Exception as e:
            print(f"{Fore.RED}Error generating signals: {str(e)}")
            return {
                'action': 'SKIP',
                'reason': f"Error: {str(e)}",
                'confidence': 0.0
            }

    def _generate_position_management_signal(self) -> Dict:
        """
        Generate signals for managing existing position

        Returns:
            Signal dictionary
        """
        pnl = self.calculate_pnl()
        pnl_percent = self.calculate_pnl_percent()

        # Check profit target
        profit_target = TRADING_PARAMS['profit_target']
        if pnl_percent >= profit_target:
            return {
                'action': 'EXIT',
                'reason': f"Profit target reached ({pnl_percent:.1%} >= {profit_target:.1%})",
                'confidence': 1.0,
                'pnl': pnl
            }

        # Check stop loss
        stop_loss = TRADING_PARAMS['stop_loss']
        if pnl_percent <= -stop_loss:
            return {
                'action': 'EXIT',
                'reason': f"Stop loss hit ({pnl_percent:.1%} <= {-stop_loss:.1%})",
                'confidence': 1.0,
                'pnl': pnl
            }

        # Check delta for hedging
        portfolio_delta = self.calculate_portfolio_delta()
        if abs(portfolio_delta) > TRADING_PARAMS['delta_threshold']:
            return {
                'action': 'HEDGE',
                'reason': f"Delta out of range ({portfolio_delta:.1f})",
                'confidence': 0.8,
                'hedge_direction': 'BUY' if portfolio_delta < 0 else 'SELL',
                'hedge_quantity': abs(portfolio_delta) * TRADING_PARAMS['delta_adjustment_size']
            }

        # Check for expiry
        if self._is_near_expiry():
            return {
                'action': 'EXIT',
                'reason': "Near expiry - theta risk",
                'confidence': 0.9,
                'pnl': pnl
            }

        # Hold position
        return {
            'action': 'HOLD',
            'reason': f"Position healthy - P&L: {pnl_percent:.1%}",
            'confidence': 0.7,
            'pnl': pnl,
            'delta': portfolio_delta
        }

    def calculate_portfolio_delta(self) -> float:
        """Calculate total portfolio delta"""
        if not self.current_position:
            return 0.0

        try:
            chain = self.get_option_chain()
            ce_strike = self.current_position.get('ce_strike')
            pe_strike = self.current_position.get('pe_strike')
            lots = self.current_position.get('lots', 1)
            lot_size = INSTRUMENT_CONFIG['lot_size']

            ce_row = chain[chain['strike'] == ce_strike].iloc[0]
            pe_row = chain[chain['strike'] == pe_strike].iloc[0]

            # For short strangle: delta = -(CE delta) - (PE delta)
            # For long strangle: delta = +(CE delta) + (PE delta)
            position_type = self.current_position.get('type', 'SHORT_STRANGLE')

            if position_type == 'SHORT_STRANGLE':
                delta = -(ce_row['ce_delta'] + pe_row['pe_delta']) * lots * lot_size
            else:
                delta = (ce_row['ce_delta'] + pe_row['pe_delta']) * lots * lot_size

            return delta

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not calculate delta: {str(e)}")
            return 0.0

    def calculate_pnl(self) -> float:
        """Calculate current P&L in rupees"""
        if not self.current_position:
            return 0.0

        try:
            chain = self.get_option_chain()
            ce_strike = self.current_position.get('ce_strike')
            pe_strike = self.current_position.get('pe_strike')
            lots = self.current_position.get('lots', 1)
            lot_size = INSTRUMENT_CONFIG['lot_size']

            ce_row = chain[chain['strike'] == ce_strike].iloc[0]
            pe_row = chain[chain['strike'] == pe_strike].iloc[0]

            entry_ce_premium = self.current_position.get('ce_entry_premium', 0)
            entry_pe_premium = self.current_position.get('pe_entry_premium', 0)

            current_ce_premium = ce_row['ce_ltp']
            current_pe_premium = pe_row['pe_ltp']

            position_type = self.current_position.get('type', 'SHORT_STRANGLE')

            if position_type == 'SHORT_STRANGLE':
                # Sold options - profit when premium decreases
                pnl = ((entry_ce_premium - current_ce_premium) +
                       (entry_pe_premium - current_pe_premium)) * lots * lot_size
            else:
                # Bought options - profit when premium increases
                pnl = ((current_ce_premium - entry_ce_premium) +
                       (current_pe_premium - entry_pe_premium)) * lots * lot_size

            return pnl

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not calculate P&L: {str(e)}")
            return 0.0

    def calculate_pnl_percent(self) -> float:
        """Calculate P&L as percentage of premium collected/paid"""
        if not self.current_position:
            return 0.0

        try:
            lots = self.current_position.get('lots', 1)
            lot_size = INSTRUMENT_CONFIG['lot_size']

            entry_ce_premium = self.current_position.get('ce_entry_premium', 0)
            entry_pe_premium = self.current_position.get('pe_entry_premium', 0)

            total_entry_premium = (entry_ce_premium + entry_pe_premium) * lots * lot_size

            if total_entry_premium == 0:
                return 0.0

            pnl = self.calculate_pnl()
            return pnl / total_entry_premium

        except Exception as e:
            return 0.0

    def _calculate_hedge(self) -> Dict:
        """Calculate hedge parameters"""
        return {
            'instrument': TRADING_PARAMS['hedge_instrument'],
            'quantity': TRADING_PARAMS['hedge_ratio'],
            'direction': 'BUY'
        }

    def _is_near_expiry(self) -> bool:
        """Check if position is near expiry"""
        if not self.entry_time:
            return False

        # Check if it's expiry day (Wednesday after 2 PM)
        now = datetime.now(self.ist)
        if now.weekday() == 2 and now.hour >= 14:  # Wednesday after 2 PM
            return True

        return False


if __name__ == "__main__":
    print("Testing Trading Strategy...")
    # This would require a Kite instance for full testing
    print("Strategy module loaded successfully")
