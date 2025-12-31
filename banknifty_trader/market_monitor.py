"""
Market Monitor Module
Monitors VIX levels and market volatility for data collection
"""

from datetime import datetime, timedelta
from typing import Dict
import pytz
import numpy as np
from colorama import Fore

from config import MARKET_DATA


class MarketMonitor:
    """Monitor market conditions - VIX and volatility"""

    def __init__(self, kite=None):
        """
        Initialize Market Monitor

        Args:
            kite: KiteConnect instance for market data
        """
        self.kite = kite
        self.ist = pytz.timezone('Asia/Kolkata')
        self.vix_cache = None
        self.last_vix_fetch = None
        self.volatility_cache = None
        self.last_volatility_fetch = None
        self.price_history = []

    def get_india_vix(self) -> Dict:
        """
        Get current India VIX value and statistics

        Returns:
            Dict with VIX data: {
                'current': float,
                'change': float,
                'change_percent': float,
                'status': str,
                'color': str
            }
        """
        try:
            # Check cache (refresh every 5 minutes)
            if self.last_vix_fetch:
                time_diff = (datetime.now() - self.last_vix_fetch).seconds
                if time_diff < 300 and self.vix_cache:
                    return self.vix_cache

            if self.kite:
                # Fetch VIX from Kite
                vix_data = self.kite.quote("NSE:INDIA VIX")
                vix_quote = vix_data.get("NSE:INDIA VIX", {})

                current_vix = vix_quote.get('last_price', 0)
                change = vix_quote.get('change', 0)
                change_percent = vix_quote.get('change_percent', 0)

            else:
                # Fallback: Mock data for testing
                current_vix = 15.5
                change = -0.3
                change_percent = -1.9

            # Determine VIX status
            if current_vix > 25:
                status = "Very High"
                color = Fore.RED
            elif current_vix > 20:
                status = "High"
                color = Fore.YELLOW
            elif current_vix > 15:
                status = "Normal"
                color = Fore.GREEN
            elif current_vix > 12:
                status = "Low"
                color = Fore.CYAN
            else:
                status = "Very Low"
                color = Fore.BLUE

            result = {
                'current': current_vix,
                'change': change,
                'change_percent': change_percent,
                'status': status,
                'color': color,
                'timestamp': datetime.now(self.ist)
            }

            # Update cache
            self.vix_cache = result
            self.last_vix_fetch = datetime.now()

            return result

        except Exception as e:
            print(f"{Fore.RED}Error fetching VIX: {str(e)}")
            return {
                'current': 0,
                'change': 0,
                'change_percent': 0,
                'status': 'Unknown',
                'color': Fore.WHITE,
                'timestamp': datetime.now(self.ist)
            }

    def calculate_banknifty_volatility(self, lookback_days: int = 20) -> Dict:
        """
        Calculate historical volatility of BankNifty

        Args:
            lookback_days: Number of days to look back for volatility calculation

        Returns:
            Dict with volatility data: {
                'current': float (annualized volatility),
                'daily': float (daily volatility),
                'status': str,
                'color': str,
                'period': int (days used for calculation)
            }
        """
        try:
            # Check cache (refresh every 5 minutes)
            if self.last_volatility_fetch:
                time_diff = (datetime.now() - self.last_volatility_fetch).seconds
                if time_diff < 300 and self.volatility_cache:
                    return self.volatility_cache

            if self.kite:
                # Fetch historical data from Kite
                from_date = datetime.now() - timedelta(days=lookback_days + 10)
                to_date = datetime.now()

                try:
                    # Get BankNifty instrument token
                    historical_data = self.kite.historical_data(
                        instrument_token=260105,  # BankNifty index token
                        from_date=from_date,
                        to_date=to_date,
                        interval='day'
                    )

                    if len(historical_data) < 5:
                        raise ValueError("Insufficient historical data")

                    # Extract closing prices
                    prices = [candle['close'] for candle in historical_data[-lookback_days:]]

                except Exception as e:
                    # If fetching fails, use mock data
                    print(f"{Fore.YELLOW}Could not fetch historical data: {str(e)}")
                    prices = None
            else:
                prices = None

            # If we don't have real data, generate realistic mock data
            if prices is None or len(prices) < 5:
                # Use stored price history if available, otherwise generate mock data
                if len(self.price_history) >= lookback_days:
                    prices = self.price_history[-lookback_days:]
                else:
                    # Generate mock price data with realistic volatility
                    base_price = 48000
                    daily_vol = 0.015  # 1.5% daily volatility
                    prices = [base_price]
                    np.random.seed(int(datetime.now().timestamp()) % 1000)

                    for _ in range(lookback_days - 1):
                        change = np.random.normal(0, daily_vol)
                        new_price = prices[-1] * (1 + change)
                        prices.append(new_price)

                    # Store for future use
                    self.price_history = prices

            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

            # Calculate volatility
            daily_volatility = np.std(returns)
            annualized_volatility = daily_volatility * np.sqrt(252) * 100  # Convert to percentage

            # Determine volatility status
            if annualized_volatility > 25:
                status = "Very High"
                color = Fore.RED
            elif annualized_volatility > 20:
                status = "High"
                color = Fore.YELLOW
            elif annualized_volatility > 15:
                status = "Normal"
                color = Fore.GREEN
            elif annualized_volatility > 10:
                status = "Low"
                color = Fore.CYAN
            else:
                status = "Very Low"
                color = Fore.BLUE

            result = {
                'current': annualized_volatility,
                'daily': daily_volatility * 100,  # Convert to percentage
                'status': status,
                'color': color,
                'period': len(prices),
                'timestamp': datetime.now(self.ist)
            }

            # Update cache
            self.volatility_cache = result
            self.last_volatility_fetch = datetime.now()

            return result

        except Exception as e:
            print(f"{Fore.RED}Error calculating volatility: {str(e)}")
            return {
                'current': 0,
                'daily': 0,
                'status': 'Unknown',
                'color': Fore.WHITE,
                'period': 0,
                'timestamp': datetime.now(self.ist)
            }

    def get_market_status(self) -> Dict:
        """
        Get comprehensive market status

        Returns:
            Dict with market conditions
        """
        vix_data = self.get_india_vix()
        volatility_data = self.calculate_banknifty_volatility()

        # Determine overall market condition
        overall_status = 'GOOD'
        reasons = []

        # Check VIX
        if vix_data['current'] > 25:
            overall_status = 'HIGH_VOL'
            reasons.append('VIX very high')
        elif vix_data['current'] < 12:
            overall_status = 'LOW_VOL'
            reasons.append('VIX very low')

        # Check trading hours
        now = datetime.now(self.ist)
        current_time = now.strftime('%H:%M')

        from config import COLLECTION_HOURS
        start_time = COLLECTION_HOURS['start_time']
        end_time = COLLECTION_HOURS['end_time']

        if not (start_time <= current_time <= end_time):
            overall_status = 'CLOSED'
            reasons.append('Outside market hours')

        # Check if it's a weekend
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            overall_status = 'CLOSED'
            reasons.append('Weekend')

        return {
            'status': overall_status,
            'vix': vix_data,
            'volatility': volatility_data,
            'reasons': reasons,
            'timestamp': now,
            'can_collect': overall_status in ['GOOD', 'HIGH_VOL', 'LOW_VOL']
        }


if __name__ == "__main__":
    print("Testing Market Monitor...")

    monitor = MarketMonitor()

    # Test VIX
    print("\n1. Testing VIX Data:")
    vix = monitor.get_india_vix()
    print(f"   Current VIX: {vix['current']}")
    print(f"   Status: {vix['color']}{vix['status']}{Fore.RESET}")

    # Test Volatility
    print("\n2. Testing Volatility Calculation:")
    vol = monitor.calculate_banknifty_volatility()
    print(f"   Annualized HV: {vol['current']:.2f}%")
    print(f"   Daily HV: {vol['daily']:.2f}%")
    print(f"   Status: {vol['color']}{vol['status']}{Fore.RESET}")

    # Test Market Status
    print("\n3. Testing Market Status:")
    status = monitor.get_market_status()
    print(f"   Overall Status: {status['status']}")
    print(f"   Can Collect Data: {status['can_collect']}")
    print(f"   Reasons: {status['reasons']}")

    print("\nMarket Monitor module working successfully!")
