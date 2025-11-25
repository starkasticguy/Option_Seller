"""
Option Pricing Simulator
Implements Black-Scholes option pricing model with Greeks calculation
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from backtest_config import OPTION_PRICING_CONFIG, STRATEGY_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlackScholesModel:
    """
    Black-Scholes option pricing model with Greeks calculation
    """

    def __init__(
        self,
        risk_free_rate: float = None,
        dividend_yield: float = None
    ):
        """
        Initialize Black-Scholes model

        Args:
            risk_free_rate: Annual risk-free rate (default from config)
            dividend_yield: Annual dividend yield (default from config)
        """
        self.r = risk_free_rate or OPTION_PRICING_CONFIG['risk_free_rate']
        self.q = dividend_yield or OPTION_PRICING_CONFIG['dividend_yield']

    def d1(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """
        Calculate d1 parameter for Black-Scholes

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)

        Returns:
            d1 value
        """
        if T <= 0:
            return 0.0

        numerator = np.log(S / K) + (self.r - self.q + 0.5 * sigma ** 2) * T
        denominator = sigma * np.sqrt(T)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def d2(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """
        Calculate d2 parameter for Black-Scholes

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)

        Returns:
            d2 value
        """
        return self.d1(S, K, T, sigma) - sigma * np.sqrt(T)

    def call_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """
        Calculate European call option price

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)

        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)

        d1_val = self.d1(S, K, T, sigma)
        d2_val = self.d2(S, K, T, sigma)

        call = (S * np.exp(-self.q * T) * norm.cdf(d1_val) -
                K * np.exp(-self.r * T) * norm.cdf(d2_val))

        return max(call, 0)

    def put_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """
        Calculate European put option price

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (in years)
            sigma: Volatility (annualized)

        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)

        d1_val = self.d1(S, K, T, sigma)
        d2_val = self.d2(S, K, T, sigma)

        put = (K * np.exp(-self.r * T) * norm.cdf(-d2_val) -
               S * np.exp(-self.q * T) * norm.cdf(-d1_val))

        return max(put, 0)

    def delta_call(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate call delta"""
        if T <= 0:
            return 1.0 if S > K else 0.0

        d1_val = self.d1(S, K, T, sigma)
        return np.exp(-self.q * T) * norm.cdf(d1_val)

    def delta_put(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate put delta"""
        if T <= 0:
            return -1.0 if S < K else 0.0

        d1_val = self.d1(S, K, T, sigma)
        return np.exp(-self.q * T) * (norm.cdf(d1_val) - 1)

    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate gamma (same for call and put)"""
        if T <= 0 or sigma == 0:
            return 0.0

        d1_val = self.d1(S, K, T, sigma)
        gamma = (np.exp(-self.q * T) * norm.pdf(d1_val)) / (S * sigma * np.sqrt(T))

        return gamma

    def theta_call(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate call theta (per day)"""
        if T <= 0:
            return 0.0

        d1_val = self.d1(S, K, T, sigma)
        d2_val = self.d2(S, K, T, sigma)

        term1 = -(S * np.exp(-self.q * T) * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
        term2 = self.q * S * np.exp(-self.q * T) * norm.cdf(d1_val)
        term3 = -self.r * K * np.exp(-self.r * T) * norm.cdf(d2_val)

        theta_annual = term1 + term2 + term3

        # Convert to per-day theta
        return theta_annual / 365.0

    def theta_put(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate put theta (per day)"""
        if T <= 0:
            return 0.0

        d1_val = self.d1(S, K, T, sigma)
        d2_val = self.d2(S, K, T, sigma)

        term1 = -(S * np.exp(-self.q * T) * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
        term2 = -self.q * S * np.exp(-self.q * T) * norm.cdf(-d1_val)
        term3 = self.r * K * np.exp(-self.r * T) * norm.cdf(-d2_val)

        theta_annual = term1 + term2 + term3

        # Convert to per-day theta
        return theta_annual / 365.0

    def vega(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate vega (same for call and put, per 1% change in volatility)"""
        if T <= 0:
            return 0.0

        d1_val = self.d1(S, K, T, sigma)
        vega = S * np.exp(-self.q * T) * norm.pdf(d1_val) * np.sqrt(T)

        # Return vega per 1% change in volatility
        return vega / 100.0

    def rho_call(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate call rho (per 1% change in interest rate)"""
        if T <= 0:
            return 0.0

        d2_val = self.d2(S, K, T, sigma)
        rho = K * T * np.exp(-self.r * T) * norm.cdf(d2_val)

        # Return rho per 1% change in interest rate
        return rho / 100.0

    def rho_put(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """Calculate put rho (per 1% change in interest rate)"""
        if T <= 0:
            return 0.0

        d2_val = self.d2(S, K, T, sigma)
        rho = -K * T * np.exp(-self.r * T) * norm.cdf(-d2_val)

        # Return rho per 1% change in interest rate
        return rho / 100.0


class OptionChainSimulator:
    """
    Simulates entire option chains with volatility smile/skew
    """

    def __init__(self, bs_model: BlackScholesModel = None):
        """
        Initialize option chain simulator

        Args:
            bs_model: Black-Scholes model instance (creates new if None)
        """
        self.bs_model = bs_model or BlackScholesModel()
        self.config = OPTION_PRICING_CONFIG

    def calculate_skewed_iv(
        self,
        strike: float,
        spot: float,
        atm_iv: float,
        option_type: str
    ) -> float:
        """
        Calculate IV with volatility skew

        Args:
            strike: Option strike price
            spot: Current spot price
            atm_iv: ATM implied volatility
            option_type: 'CE' or 'PE'

        Returns:
            Adjusted IV with skew
        """
        if not self.config['apply_skew']:
            return atm_iv

        # Calculate moneyness
        moneyness = np.log(strike / spot)

        # Apply skew based on option type
        if option_type == 'PE':
            skew_factor = self.config['skew_factor_puts']
        else:  # CE
            skew_factor = self.config['skew_factor_calls']

        # IV increases for OTM options
        iv_adjustment = skew_factor * moneyness
        skewed_iv = atm_iv * (1 + iv_adjustment)

        # Ensure IV is reasonable (min 5%, max 100%)
        skewed_iv = max(0.05, min(1.0, skewed_iv))

        return skewed_iv

    def calculate_bid_ask_spread(
        self,
        strike: float,
        spot: float,
        price: float
    ) -> Tuple[float, float]:
        """
        Calculate realistic bid-ask spread

        Args:
            strike: Option strike price
            spot: Current spot price
            price: Theoretical option price

        Returns:
            Tuple of (bid, ask) prices
        """
        if not self.config['simulate_spreads']:
            return price, price

        # Calculate moneyness
        moneyness = abs(strike - spot) / spot

        # Determine spread factor based on moneyness
        if moneyness < 0.02:  # ATM
            spread = self.config['spread_atm']
        elif moneyness < self.config['otm_threshold']:  # OTM
            spread = self.config['spread_atm'] * self.config['spread_otm_factor']
        else:  # Deep OTM
            spread = self.config['spread_atm'] * self.config['spread_deep_otm_factor']

        # Adjust spread based on price (lower prices have wider spreads relatively)
        if price < 10:
            spread *= 1.5
        elif price < 5:
            spread *= 2.0

        half_spread = spread / 2.0
        bid = max(0, price - half_spread)
        ask = price + half_spread

        return bid, ask

    def generate_option_chain(
        self,
        spot: float,
        atm_iv: float,
        time_to_expiry: float,
        strikes: List[float] = None
    ) -> pd.DataFrame:
        """
        Generate complete option chain for given parameters

        Args:
            spot: Current spot price
            atm_iv: ATM implied volatility (annualized)
            time_to_expiry: Time to expiry in years
            strikes: List of strikes (auto-generated if None)

        Returns:
            DataFrame with option chain data
        """
        # Generate strikes if not provided
        if strikes is None:
            strikes = self._generate_strikes(spot)

        chain_data = []

        for strike in strikes:
            # Calculate skewed IVs
            ce_iv = self.calculate_skewed_iv(strike, spot, atm_iv, 'CE')
            pe_iv = self.calculate_skewed_iv(strike, spot, atm_iv, 'PE')

            # Calculate option prices
            ce_price = self.bs_model.call_price(spot, strike, time_to_expiry, ce_iv)
            pe_price = self.bs_model.put_price(spot, strike, time_to_expiry, pe_iv)

            # Calculate Greeks for CE
            ce_delta = self.bs_model.delta_call(spot, strike, time_to_expiry, ce_iv)
            ce_gamma = self.bs_model.gamma(spot, strike, time_to_expiry, ce_iv)
            ce_theta = self.bs_model.theta_call(spot, strike, time_to_expiry, ce_iv)
            ce_vega = self.bs_model.vega(spot, strike, time_to_expiry, ce_iv)

            # Calculate Greeks for PE
            pe_delta = self.bs_model.delta_put(spot, strike, time_to_expiry, pe_iv)
            pe_gamma = self.bs_model.gamma(spot, strike, time_to_expiry, pe_iv)
            pe_theta = self.bs_model.theta_put(spot, strike, time_to_expiry, pe_iv)
            pe_vega = self.bs_model.vega(spot, strike, time_to_expiry, pe_iv)

            # Calculate bid-ask spreads
            ce_bid, ce_ask = self.calculate_bid_ask_spread(strike, spot, ce_price)
            pe_bid, pe_ask = self.calculate_bid_ask_spread(strike, spot, pe_price)

            chain_data.append({
                'strike': strike,
                'ce_ltp': ce_price,
                'ce_bid': ce_bid,
                'ce_ask': ce_ask,
                'ce_iv': ce_iv,
                'ce_delta': ce_delta,
                'ce_gamma': ce_gamma,
                'ce_theta': ce_theta,
                'ce_vega': ce_vega,
                'pe_ltp': pe_price,
                'pe_bid': pe_bid,
                'pe_ask': pe_ask,
                'pe_iv': pe_iv,
                'pe_delta': pe_delta,
                'pe_gamma': pe_gamma,
                'pe_theta': pe_theta,
                'pe_vega': pe_vega,
            })

        df = pd.DataFrame(chain_data)

        # Add ATM flag
        df['atm_distance'] = abs(df['strike'] - spot)
        df['is_atm'] = df['atm_distance'] == df['atm_distance'].min()

        return df

    def _generate_strikes(self, spot: float) -> List[float]:
        """
        Generate strikes around spot price

        Args:
            spot: Current spot price

        Returns:
            List of strike prices
        """
        strike_interval = self.config['strike_interval']
        strike_range = self.config['strikes_range']

        # Round spot to nearest strike interval
        atm_strike = round(spot / strike_interval) * strike_interval

        # Generate strikes
        lower_bound = atm_strike - strike_range
        upper_bound = atm_strike + strike_range

        strikes = list(range(
            int(lower_bound),
            int(upper_bound) + strike_interval,
            strike_interval
        ))

        return strikes

    def get_option_price(
        self,
        strike: float,
        spot: float,
        atm_iv: float,
        time_to_expiry: float,
        option_type: str,
        price_type: str = 'ltp'
    ) -> float:
        """
        Get option price for a specific strike

        Args:
            strike: Strike price
            spot: Spot price
            atm_iv: ATM implied volatility
            time_to_expiry: Time to expiry in years
            option_type: 'CE' or 'PE'
            price_type: 'ltp', 'bid', or 'ask'

        Returns:
            Option price
        """
        # Calculate skewed IV
        iv = self.calculate_skewed_iv(strike, spot, atm_iv, option_type)

        # Calculate price
        if option_type == 'CE':
            price = self.bs_model.call_price(spot, strike, time_to_expiry, iv)
        else:
            price = self.bs_model.put_price(spot, strike, time_to_expiry, iv)

        # Apply bid-ask if needed
        if price_type == 'bid':
            bid, _ = self.calculate_bid_ask_spread(strike, spot, price)
            return bid
        elif price_type == 'ask':
            _, ask = self.calculate_bid_ask_spread(strike, spot, price)
            return ask
        else:
            return price


class VolatilityCalculator:
    """
    Calculate realized volatility from historical prices
    """

    @staticmethod
    def calculate_realized_volatility(
        prices: pd.Series,
        window: int = 20,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate realized volatility using log returns

        Args:
            prices: Series of prices
            window: Rolling window size
            trading_days: Trading days per year (252 for stocks, 365 for crypto)

        Returns:
            Series of annualized realized volatility
        """
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))

        # Calculate rolling standard deviation
        rolling_std = log_returns.rolling(window=window).std()

        # Annualize
        annualized_vol = rolling_std * np.sqrt(trading_days)

        return annualized_vol

    @staticmethod
    def calculate_parkinson_volatility(
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate Parkinson volatility (uses high-low range)

        Args:
            high: Series of high prices
            low: Series of low prices
            window: Rolling window size
            trading_days: Trading days per year

        Returns:
            Series of annualized Parkinson volatility
        """
        # Parkinson estimator
        hl_ratio = np.log(high / low)
        parkinson_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)

        # Rolling mean
        rolling_var = parkinson_var.rolling(window=window).mean()

        # Annualize
        annualized_vol = np.sqrt(rolling_var * trading_days)

        return annualized_vol

    @staticmethod
    def calculate_garman_klass_volatility(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility (uses OHLC)

        Args:
            open_: Series of open prices
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            window: Rolling window size
            trading_days: Trading days per year

        Returns:
            Series of annualized Garman-Klass volatility
        """
        # Garman-Klass estimator
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)

        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)

        # Rolling mean
        rolling_var = gk_var.rolling(window=window).mean()

        # Annualize
        annualized_vol = np.sqrt(rolling_var * trading_days)

        return annualized_vol


def test_option_simulator():
    """Test the option simulator"""
    print("Testing Option Simulator...")
    print("=" * 60)

    # Initialize
    bs_model = BlackScholesModel()
    simulator = OptionChainSimulator(bs_model)

    # Test parameters
    spot = 48000
    atm_iv = 0.18  # 18% IV
    time_to_expiry = 3 / 365  # 3 days to expiry

    print(f"\nTest Parameters:")
    print(f"  Spot: ₹{spot:,.0f}")
    print(f"  ATM IV: {atm_iv*100:.1f}%")
    print(f"  Time to Expiry: {time_to_expiry*365:.1f} days")

    # Generate option chain
    print("\n Generating option chain...")
    chain = simulator.generate_option_chain(spot, atm_iv, time_to_expiry)

    # Display ATM and nearby strikes
    print("\nOption Chain (ATM ± 300):")
    print("-" * 100)
    atm_strikes = chain[abs(chain['strike'] - spot) <= 300].copy()
    atm_strikes = atm_strikes.sort_values('strike')

    for _, row in atm_strikes.iterrows():
        print(f"Strike: {row['strike']:.0f}")
        print(f"  CE: Price={row['ce_ltp']:.2f} (bid={row['ce_bid']:.2f}, ask={row['ce_ask']:.2f}), "
              f"Delta={row['ce_delta']:.3f}, Theta={row['ce_theta']:.2f}, IV={row['ce_iv']*100:.1f}%")
        print(f"  PE: Price={row['pe_ltp']:.2f} (bid={row['pe_bid']:.2f}, ask={row['pe_ask']:.2f}), "
              f"Delta={row['pe_delta']:.3f}, Theta={row['pe_theta']:.2f}, IV={row['pe_iv']*100:.1f}%")
        print()

    print("\n✅ Option simulator test completed!")


if __name__ == "__main__":
    test_option_simulator()
