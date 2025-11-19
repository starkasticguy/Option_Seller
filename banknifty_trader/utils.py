"""
Utility Functions
Helper functions for the trading system
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pytz
import math


class OptionsCalculator:
    """Options pricing and greeks calculations"""

    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Call option price using Black-Scholes

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility (IV)

        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        from scipy.stats import norm
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        return call_price

    @staticmethod
    def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Put option price using Black-Scholes

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility (IV)

        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        from scipy.stats import norm
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return put_price

    @staticmethod
    def calculate_delta(S: float, K: float, T: float, r: float, sigma: float,
                       option_type: str = 'call') -> float:
        """Calculate option delta"""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        from scipy.stats import norm
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return -norm.cdf(-d1)

    @staticmethod
    def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma"""
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        from scipy.stats import norm
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        return gamma

    @staticmethod
    def calculate_theta(S: float, K: float, T: float, r: float, sigma: float,
                       option_type: str = 'call') -> float:
        """Calculate option theta (per day)"""
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        from scipy.stats import norm

        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * norm.cdf(-d2))

        return theta / 365  # Convert to daily theta

    @staticmethod
    def calculate_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega"""
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        from scipy.stats import norm
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in IV

        return vega


class TimeUtils:
    """Time and date utilities"""

    IST = pytz.timezone('Asia/Kolkata')

    @staticmethod
    def get_ist_now() -> datetime:
        """Get current IST time"""
        return datetime.now(TimeUtils.IST)

    @staticmethod
    def is_trading_hour(current_time: datetime = None) -> bool:
        """Check if current time is within trading hours"""
        from config import TRADING_PARAMS

        if current_time is None:
            current_time = TimeUtils.get_ist_now()

        start_time, end_time = TRADING_PARAMS['trading_hours']
        current_time_str = current_time.strftime('%H:%M')

        return start_time <= current_time_str <= end_time

    @staticmethod
    def is_weekday(current_time: datetime = None) -> bool:
        """Check if current day is a weekday"""
        if current_time is None:
            current_time = TimeUtils.get_ist_now()

        return current_time.weekday() < 5  # Monday=0, Friday=4

    @staticmethod
    def get_next_expiry(current_time: datetime = None) -> datetime:
        """Get next weekly expiry (Wednesday)"""
        if current_time is None:
            current_time = TimeUtils.get_ist_now()

        days_until_wednesday = (2 - current_time.weekday()) % 7

        if days_until_wednesday == 0 and current_time.hour >= 15:
            days_until_wednesday = 7

        expiry_date = current_time + timedelta(days=days_until_wednesday)
        return expiry_date.replace(hour=15, minute=30, second=0, microsecond=0)

    @staticmethod
    def days_to_expiry(current_time: datetime = None) -> float:
        """Calculate days to next expiry"""
        if current_time is None:
            current_time = TimeUtils.get_ist_now()

        expiry = TimeUtils.get_next_expiry(current_time)
        delta = expiry - current_time

        return delta.total_seconds() / (24 * 3600)


class StatisticsUtils:
    """Statistical calculations"""

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.06) -> float:
        """
        Calculate Sharpe Ratio

        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        sharpe_annual = sharpe * np.sqrt(252)  # Annualized

        return sharpe_annual

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: List[float]) -> float:
        """
        Calculate maximum drawdown

        Args:
            cumulative_returns: List of cumulative returns

        Returns:
            Maximum drawdown (as positive value)
        """
        if not cumulative_returns:
            return 0.0

        cumulative = np.array(cumulative_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown)

        return max_dd

    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.06) -> float:
        """
        Calculate Sortino Ratio (like Sharpe but only considers downside volatility)

        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)

        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_std
        sortino_annual = sortino * np.sqrt(252)

        return sortino_annual


class FormattingUtils:
    """Display formatting utilities"""

    @staticmethod
    def format_currency(amount: float, decimals: int = 2) -> str:
        """Format amount as currency"""
        return f"₹{amount:,.{decimals}f}"

    @staticmethod
    def format_percentage(value: float, decimals: int = 2, show_sign: bool = True) -> str:
        """Format value as percentage"""
        sign = '+' if value >= 0 and show_sign else ''
        return f"{sign}{value:.{decimals}f}%"

    @staticmethod
    def format_greeks(greeks: Dict) -> str:
        """Format greeks dictionary as string"""
        return (f"Δ:{greeks.get('delta', 0):+.2f}, "
                f"Γ:{greeks.get('gamma', 0):.4f}, "
                f"Θ:{greeks.get('theta', 0):+.2f}, "
                f"ν:{greeks.get('vega', 0):+.2f}")

    @staticmethod
    def truncate_string(s: str, max_length: int = 50) -> str:
        """Truncate string to max length"""
        if len(s) <= max_length:
            return s
        return s[:max_length-3] + '...'


class ValidationUtils:
    """Input validation utilities"""

    @staticmethod
    def validate_strike(strike: int) -> bool:
        """Validate strike price (must be multiple of 100)"""
        return strike > 0 and strike % 100 == 0

    @staticmethod
    def validate_premium(premium: float) -> bool:
        """Validate premium value"""
        return 0 < premium < 10000

    @staticmethod
    def validate_delta(delta: float) -> bool:
        """Validate delta value (-1 to 1)"""
        return -1 <= delta <= 1

    @staticmethod
    def validate_lots(lots: int) -> bool:
        """Validate lot size"""
        return lots > 0 and lots <= 100

    @staticmethod
    def validate_vix(vix: float) -> bool:
        """Validate VIX value"""
        return 0 < vix < 100


def test_utilities():
    """Test utility functions"""
    print("Testing utility functions...")

    # Test Options Calculator
    calc = OptionsCalculator()
    S, K, T, r, sigma = 48000, 48000, 0.0192, 0.06, 0.18  # ~7 days, 18% IV

    call_price = calc.black_scholes_call(S, K, T, r, sigma)
    put_price = calc.black_scholes_put(S, K, T, r, sigma)

    print(f"\nOptions Pricing (ATM):")
    print(f"Call Price: ₹{call_price:.2f}")
    print(f"Put Price: ₹{put_price:.2f}")

    # Test Time Utils
    print(f"\nTime Utilities:")
    print(f"IST Now: {TimeUtils.get_ist_now()}")
    print(f"Is Trading Hour: {TimeUtils.is_trading_hour()}")
    print(f"Next Expiry: {TimeUtils.get_next_expiry()}")
    print(f"Days to Expiry: {TimeUtils.days_to_expiry():.2f}")

    # Test Formatting
    print(f"\nFormatting:")
    print(f"Currency: {FormattingUtils.format_currency(123456.78)}")
    print(f"Percentage: {FormattingUtils.format_percentage(12.345)}")

    print("\nAll tests completed!")


if __name__ == "__main__":
    test_utilities()
