"""
Market Monitor Module
Monitors economic events, VIX levels, and market conditions
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pytz
from bs4 import BeautifulSoup
import json
import numpy as np
from colorama import Fore, Style

from config import EVENTS_CONFIG, MARKET_DATA


class MarketMonitor:
    """Monitor market conditions and economic events"""

    def __init__(self, kite=None):
        """
        Initialize Market Monitor

        Args:
            kite: KiteConnect instance for market data
        """
        self.kite = kite
        self.ist = pytz.timezone('Asia/Kolkata')
        self.event_cache = []
        self.last_event_fetch = None
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
                    # Format: NSE:NIFTY BANK or BANKNIFTY
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
                    # If fetching fails, use simulated data
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

    def check_economic_events(self) -> Dict:
        """
        Check for upcoming high-impact economic events

        Returns:
            Dict with event data: {
                'status': 'RED/YELLOW/GREEN',
                'risk_score': int,
                'events': List[Dict],
                'message': str
            }
        """
        try:
            # Refresh cache every 30 minutes
            if self.last_event_fetch:
                time_diff = (datetime.now() - self.last_event_fetch).seconds
                if time_diff < 1800 and self.event_cache:
                    return self._analyze_events(self.event_cache)

            # Fetch events from multiple sources
            events = []

            # Source 1: Forex Factory (US events)
            events.extend(self._fetch_forex_factory_events())

            # Source 2: MoneyControl (India events)
            events.extend(self._fetch_india_events())

            # Source 3: Hardcoded known events (fallback)
            events.extend(self._get_known_upcoming_events())

            # Update cache
            self.event_cache = events
            self.last_event_fetch = datetime.now()

            return self._analyze_events(events)

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not fetch events: {str(e)}")
            # Return safe default (assume medium risk)
            return {
                'status': 'YELLOW',
                'risk_score': 3,
                'events': [],
                'message': 'Unable to fetch events, trading with caution'
            }

    def _fetch_forex_factory_events(self) -> List[Dict]:
        """
        Fetch events from Forex Factory calendar

        Returns:
            List of event dictionaries
        """
        events = []
        try:
            # Note: This is a simplified version
            # In production, you'd need to handle AJAX/dynamic content
            # or use their API if available

            url = EVENTS_CONFIG['forex_factory_url']
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # For now, return empty (requires selenium for dynamic content)
            # TODO: Implement selenium-based scraping or use paid API

            pass

        except Exception as e:
            print(f"{Fore.YELLOW}Could not fetch Forex Factory events: {str(e)}")

        return events

    def _fetch_india_events(self) -> List[Dict]:
        """
        Fetch Indian economic events

        Returns:
            List of event dictionaries
        """
        events = []
        try:
            # This would require web scraping from MoneyControl
            # For MVP, we'll use the hardcoded approach below
            pass

        except Exception as e:
            print(f"{Fore.YELLOW}Could not fetch India events: {str(e)}")

        return events

    def _get_known_upcoming_events(self) -> List[Dict]:
        """
        Get known upcoming events (hardcoded for reliability)
        This should be manually updated with known major events

        Returns:
            List of event dictionaries
        """
        # Define known events for next 30 days
        # Format: (date_string, event_name, impact_score)
        known_events = [
            # Add known events here
            # Example:
            # ('2024-01-15 14:00', 'RBI Monetary Policy', 5),
            # ('2024-01-20 20:00', 'US FOMC Meeting', 5),
        ]

        events = []
        now = datetime.now(self.ist)

        for date_str, name, impact in known_events:
            try:
                event_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                event_time = self.ist.localize(event_time)

                # Only include events within the next 30 days
                if now <= event_time <= now + timedelta(days=30):
                    events.append({
                        'name': name,
                        'time': event_time,
                        'impact': impact,
                        'hours_until': (event_time - now).total_seconds() / 3600
                    })
            except Exception as e:
                continue

        return events

    def _analyze_events(self, events: List[Dict]) -> Dict:
        """
        Analyze events and determine risk level

        Args:
            events: List of event dictionaries

        Returns:
            Analysis result dictionary
        """
        now = datetime.now(self.ist)
        horizon_hours = EVENTS_CONFIG['event_horizon_hours']

        # Filter events in the next X hours
        upcoming_events = []
        for event in events:
            hours_until = event.get('hours_until', float('inf'))
            if 0 <= hours_until <= horizon_hours:
                upcoming_events.append(event)

        # Calculate risk score
        risk_score = 0
        high_impact_count = 0
        medium_impact_count = 0

        for event in upcoming_events:
            impact = event.get('impact', 0)
            if impact >= EVENTS_CONFIG['high_impact_threshold']:
                risk_score += impact
                high_impact_count += 1
            elif impact >= EVENTS_CONFIG['medium_impact_threshold']:
                risk_score += impact * 0.5
                medium_impact_count += 1

        # Determine status
        if high_impact_count > 0:
            status = 'RED'
            message = f"{high_impact_count} high impact event(s) in next {horizon_hours}h - SKIP TRADING"
        elif medium_impact_count > 0:
            status = 'YELLOW'
            message = f"{medium_impact_count} medium impact event(s) - REDUCE POSITION SIZE"
        else:
            status = 'GREEN'
            message = "No major events - NORMAL TRADING"

        return {
            'status': status,
            'risk_score': risk_score,
            'events': upcoming_events,
            'message': message,
            'high_impact_count': high_impact_count,
            'medium_impact_count': medium_impact_count
        }

    def get_market_status(self) -> Dict:
        """
        Get comprehensive market status

        Returns:
            Dict with market conditions
        """
        vix_data = self.get_india_vix()
        event_data = self.check_economic_events()
        volatility_data = self.calculate_banknifty_volatility()

        # Determine overall market condition
        overall_status = 'GOOD'
        reasons = []

        # Check VIX
        if vix_data['current'] > 25:
            overall_status = 'BAD'
            reasons.append('VIX too high')
        elif vix_data['current'] < 12:
            overall_status = 'CAUTION'
            reasons.append('VIX too low')

        # Check events
        if event_data['status'] == 'RED':
            overall_status = 'BAD'
            reasons.append('High impact events')
        elif event_data['status'] == 'YELLOW':
            if overall_status == 'GOOD':
                overall_status = 'CAUTION'
            reasons.append('Medium impact events')

        # Check trading hours
        now = datetime.now(self.ist)
        current_time = now.strftime('%H:%M')

        from config import TRADING_PARAMS
        start_time, end_time = TRADING_PARAMS['trading_hours']

        if not (start_time <= current_time <= end_time):
            overall_status = 'CLOSED'
            reasons.append('Outside trading hours')

        # Check if it's a weekend
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            overall_status = 'CLOSED'
            reasons.append('Weekend')

        return {
            'status': overall_status,
            'vix': vix_data,
            'volatility': volatility_data,
            'events': event_data,
            'reasons': reasons,
            'timestamp': now,
            'can_trade': overall_status in ['GOOD', 'CAUTION']
        }

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on current conditions

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        market_status = self.get_market_status()

        if market_status['status'] == 'CLOSED':
            return False, "Market closed"

        if market_status['status'] == 'BAD':
            reasons = ', '.join(market_status['reasons'])
            return False, f"Unfavorable conditions: {reasons}"

        if market_status['events']['status'] == 'RED':
            if EVENTS_CONFIG['skip_trading_on_high_impact']:
                return False, "High impact event upcoming"

        return True, "Trading allowed"

    def get_position_size_adjustment(self) -> float:
        """
        Get position size adjustment factor based on market conditions

        Returns:
            Multiplier for position size (0.0 to 1.0)
        """
        market_status = self.get_market_status()

        # Start with full size
        adjustment = 1.0

        # Reduce on high VIX
        vix = market_status['vix']['current']
        if vix > 22:
            adjustment *= 0.5
        elif vix > 20:
            adjustment *= 0.75

        # Reduce on medium impact events
        if market_status['events']['status'] == 'YELLOW':
            if EVENTS_CONFIG['reduce_size_on_medium_impact']:
                adjustment *= EVENTS_CONFIG['size_reduction_factor']

        return adjustment

    def display_event_summary(self):
        """Display a formatted summary of upcoming events"""
        event_data = self.check_economic_events()

        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}ECONOMIC EVENT SUMMARY")
        print(f"{Fore.CYAN}{'='*60}")

        status = event_data['status']
        if status == 'RED':
            color = Fore.RED
            symbol = '🔴'
        elif status == 'YELLOW':
            color = Fore.YELLOW
            symbol = '🟡'
        else:
            color = Fore.GREEN
            symbol = '🟢'

        print(f"{color}{symbol} Status: {status}")
        print(f"{Fore.WHITE}Message: {event_data['message']}")
        print(f"Risk Score: {event_data['risk_score']}")

        if event_data['events']:
            print(f"\n{Fore.YELLOW}Upcoming Events:")
            for event in event_data['events'][:5]:  # Show top 5
                impact = event['impact']
                impact_str = '⚠️ HIGH' if impact >= 4 else '⚡ MED'
                hours = event.get('hours_until', 0)
                print(f"  {impact_str} | {event['name']} in {hours:.1f}h")
        else:
            print(f"\n{Fore.GREEN}✓ No major events in next 24 hours")

        print(f"{Fore.CYAN}{'='*60}\n")


def test_market_monitor():
    """Test the market monitor functionality"""
    print("Testing Market Monitor...")

    monitor = MarketMonitor()

    # Test VIX
    print("\n1. Testing VIX Data:")
    vix = monitor.get_india_vix()
    print(f"   Current VIX: {vix['current']}")
    print(f"   Status: {vix['color']}{vix['status']}{Style.RESET_ALL}")

    # Test Events
    print("\n2. Testing Event Detection:")
    events = monitor.check_economic_events()
    print(f"   Status: {events['status']}")
    print(f"   Message: {events['message']}")
    print(f"   Events found: {len(events['events'])}")

    # Test Market Status
    print("\n3. Testing Market Status:")
    status = monitor.get_market_status()
    print(f"   Overall Status: {status['status']}")
    print(f"   Can Trade: {status['can_trade']}")
    print(f"   Reasons: {status['reasons']}")

    # Test Trading Allowed
    print("\n4. Testing Trading Permission:")
    allowed, reason = monitor.is_trading_allowed()
    print(f"   Allowed: {allowed}")
    print(f"   Reason: {reason}")

    # Test Position Sizing
    print("\n5. Testing Position Size Adjustment:")
    adjustment = monitor.get_position_size_adjustment()
    print(f"   Size Multiplier: {adjustment:.2f}")

    # Display Summary
    monitor.display_event_summary()


if __name__ == "__main__":
    test_market_monitor()
