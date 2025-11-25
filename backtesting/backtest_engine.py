"""
Backtesting Engine
Main engine for running backtests of BankNifty options strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum

from backtest_config import (
    STRATEGY_CONFIG, COST_CONFIG, EVENTS_CONFIG,
    get_cost_per_trade
)
from option_simulator import OptionChainSimulator, BlackScholesModel, VolatilityCalculator
from data_collector import HistoricalDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Position type enumeration"""
    SHORT_STRANGLE = "SHORT_STRANGLE"
    LONG_STRANGLE = "LONG_STRANGLE"
    IRON_CONDOR = "IRON_CONDOR"


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Position:
    """Represents an options position"""
    entry_time: pd.Timestamp
    position_type: PositionType
    ce_strike: float
    pe_strike: float
    ce_entry_price: float
    pe_entry_price: float
    lots: int
    lot_size: int

    # Exit info (filled when closed)
    exit_time: Optional[pd.Timestamp] = None
    ce_exit_price: float = 0.0
    pe_exit_price: float = 0.0
    exit_reason: str = ""
    status: PositionStatus = PositionStatus.OPEN

    # Greeks at entry
    ce_delta: float = 0.0
    pe_delta: float = 0.0
    ce_theta: float = 0.0
    pe_theta: float = 0.0
    ce_vega: float = 0.0
    pe_vega: float = 0.0

    # Transaction costs
    entry_costs: float = 0.0
    exit_costs: float = 0.0
    slippage_costs: float = 0.0

    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0

    def calculate_pnl(
        self,
        ce_current_price: float,
        pe_current_price: float,
        include_costs: bool = True
    ) -> float:
        """Calculate current P&L"""
        if self.position_type == PositionType.SHORT_STRANGLE:
            # Sold options - profit when premium decreases
            pnl = (
                (self.ce_entry_price - ce_current_price) +
                (self.pe_entry_price - pe_current_price)
            ) * self.lots * self.lot_size
        else:  # LONG_STRANGLE
            # Bought options - profit when premium increases
            pnl = (
                (ce_current_price - self.ce_entry_price) +
                (pe_current_price - self.pe_entry_price)
            ) * self.lots * self.lot_size

        if include_costs:
            pnl -= (self.entry_costs + self.exit_costs + self.slippage_costs)

        return pnl

    def calculate_portfolio_delta(
        self,
        ce_delta: float,
        pe_delta: float
    ) -> float:
        """Calculate portfolio delta"""
        if self.position_type == PositionType.SHORT_STRANGLE:
            delta = -(ce_delta + pe_delta) * self.lots * self.lot_size
        else:
            delta = (ce_delta + pe_delta) * self.lots * self.lot_size

        return delta


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    position_type: str
    ce_strike: float
    pe_strike: float
    ce_entry_price: float
    pe_entry_price: float
    ce_exit_price: float
    pe_exit_price: float
    lots: int
    pnl: float
    pnl_percent: float
    exit_reason: str
    entry_vix: float = 0.0
    exit_vix: float = 0.0
    entry_iv: float = 0.0
    exit_iv: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    hold_time_minutes: int = 0
    transaction_costs: float = 0.0


class BacktestEngine:
    """
    Main backtesting engine for BankNifty options strategies
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        strategy_config: Dict = None,
        cost_config: Dict = None
    ):
        """
        Initialize backtest engine

        Args:
            market_data: Historical market data with OHLCV, VIX, RV
            strategy_config: Strategy configuration (default from config)
            cost_config: Cost configuration (default from config)
        """
        self.market_data = market_data
        self.strategy_config = strategy_config or STRATEGY_CONFIG
        self.cost_config = cost_config or COST_CONFIG

        # Initialize components
        self.bs_model = BlackScholesModel()
        self.option_simulator = OptionChainSimulator(self.bs_model)
        self.data_collector = HistoricalDataCollector()

        # State tracking
        self.current_position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.daily_pnl: List[Dict] = []

        # Capital management
        self.initial_capital = self.strategy_config['initial_capital']
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital

        # Get expiry dates
        start_date = market_data.index[0].strftime('%Y-%m-%d')
        end_date = market_data.index[-1].strftime('%Y-%m-%d')
        self.expiry_dates = self.data_collector.get_expiry_dates(start_date, end_date)

        # IV percentile tracking
        self.iv_history: List[float] = []

        logger.info(f"Backtest engine initialized")
        logger.info(f"  Data period: {start_date} to {end_date}")
        logger.info(f"  Records: {len(market_data)}")
        logger.info(f"  Expiry dates: {len(self.expiry_dates)}")
        logger.info(f"  Initial capital: ₹{self.initial_capital:,.0f}")

    def run(self) -> Dict:
        """
        Run the backtest

        Returns:
            Dictionary with backtest results
        """
        logger.info("=" * 60)
        logger.info("Starting backtest...")
        logger.info("=" * 60)

        # Iterate through market data
        for idx, (timestamp, row) in enumerate(self.market_data.iterrows()):
            # Update equity curve
            self._update_equity_curve(timestamp, row)

            # Skip warmup period
            if idx < self.strategy_config.get('warmup_days', 30):
                continue

            # Check if we have a position
            if self.current_position is not None:
                # Manage existing position
                self._manage_position(timestamp, row)
            else:
                # Look for entry signals
                self._check_entry_signals(timestamp, row)

            # Log progress
            if idx % 1000 == 0:
                logger.info(f"  Processed {idx}/{len(self.market_data)} bars")

        # Close any open position at end
        if self.current_position is not None:
            last_row = self.market_data.iloc[-1]
            self._exit_position(
                self.market_data.index[-1],
                last_row,
                "End of backtest"
            )

        logger.info("=" * 60)
        logger.info("✅ Backtest completed!")
        logger.info(f"  Total trades: {len(self.trades)}")
        logger.info(f"  Final capital: ₹{self.current_capital:,.0f}")
        logger.info(f"  Total return: {(self.current_capital/self.initial_capital - 1)*100:.2f}%")
        logger.info("=" * 60)

        # Compile results
        results = self._compile_results()
        return results

    def _check_entry_signals(
        self,
        timestamp: pd.Timestamp,
        market_row: pd.Series
    ) -> None:
        """Check for entry signals"""
        # Check trading hours
        if not self._is_trading_time(timestamp):
            return

        # Check if it's expiry day (might want to avoid new entries)
        if self._is_expiry_day(timestamp):
            return

        # Check VIX conditions
        vix = market_row.get('VIX', 0)
        vix_low, vix_high = self.strategy_config['vix_range']

        if vix < vix_low or vix > vix_high:
            return

        # Check IV percentile
        iv_percentile = self._calculate_iv_percentile(market_row)
        if iv_percentile < self.strategy_config['iv_percentile_min']:
            return

        # Check for major events
        if self._is_major_event_day(timestamp):
            return

        # All conditions met - enter position
        self._enter_position(timestamp, market_row)

    def _enter_position(
        self,
        timestamp: pd.Timestamp,
        market_row: pd.Series
    ) -> None:
        """Enter a new position"""
        spot = market_row['Close']
        vix = market_row.get('VIX', 18)

        # Get time to next expiry
        time_to_expiry = self._get_time_to_expiry(timestamp)

        if time_to_expiry <= 0:
            return  # No valid expiry

        # Use realized volatility or VIX
        rv_20d = market_row.get('RV_20d', vix / 100)
        atm_iv = max(rv_20d, vix / 100)  # Use higher of RV or VIX

        # Generate option chain
        chain = self.option_simulator.generate_option_chain(
            spot=spot,
            atm_iv=atm_iv,
            time_to_expiry=time_to_expiry
        )

        # Select strikes based on method
        strikes = self._select_strikes(chain, spot)

        if strikes is None:
            return

        ce_strike = strikes['CE']['strike']
        pe_strike = strikes['PE']['strike']

        # Get option prices and Greeks
        ce_row = chain[chain['strike'] == ce_strike].iloc[0]
        pe_row = chain[chain['strike'] == pe_strike].iloc[0]

        # Determine entry price based on position type
        position_type = PositionType[self.strategy_config['strategy_type']]

        if position_type == PositionType.SHORT_STRANGLE:
            # Selling - use bid price
            ce_entry_price = ce_row['ce_bid']
            pe_entry_price = pe_row['pe_bid']
            slippage = self.cost_config['entry_slippage']
            ce_entry_price -= slippage
            pe_entry_price -= slippage
        else:
            # Buying - use ask price
            ce_entry_price = ce_row['ce_ask']
            pe_entry_price = pe_row['pe_ask']
            slippage = self.cost_config['entry_slippage']
            ce_entry_price += slippage
            pe_entry_price += slippage

        # Calculate transaction costs
        total_premium = (ce_entry_price + pe_entry_price)
        lots = self.strategy_config['position_size']
        entry_costs_detail = get_cost_per_trade(total_premium, lots)
        entry_costs = entry_costs_detail['total_cost']

        # Create position
        position = Position(
            entry_time=timestamp,
            position_type=position_type,
            ce_strike=ce_strike,
            pe_strike=pe_strike,
            ce_entry_price=ce_entry_price,
            pe_entry_price=pe_entry_price,
            lots=lots,
            lot_size=self.strategy_config['lot_size'],
            ce_delta=ce_row['ce_delta'],
            pe_delta=pe_row['pe_delta'],
            ce_theta=ce_row['ce_theta'],
            pe_theta=pe_row['pe_theta'],
            ce_vega=ce_row['ce_vega'],
            pe_vega=pe_row['pe_vega'],
            entry_costs=entry_costs,
            slippage_costs=slippage * 2 * lots * self.strategy_config['lot_size']
        )

        self.current_position = position

        logger.info(f"\n{'='*60}")
        logger.info(f"ENTRY: {timestamp}")
        logger.info(f"  Type: {position_type.value}")
        logger.info(f"  Spot: ₹{spot:,.2f} | VIX: {vix:.2f}")
        logger.info(f"  CE Strike: {ce_strike} @ ₹{ce_entry_price:.2f}")
        logger.info(f"  PE Strike: {pe_strike} @ ₹{pe_entry_price:.2f}")
        logger.info(f"  Premium: ₹{total_premium * lots * self.strategy_config['lot_size']:,.2f}")
        logger.info(f"  Costs: ₹{entry_costs:.2f}")
        logger.info(f"{'='*60}\n")

    def _manage_position(
        self,
        timestamp: pd.Timestamp,
        market_row: pd.Series
    ) -> None:
        """Manage existing position"""
        if self.current_position is None:
            return

        spot = market_row['Close']
        vix = market_row.get('VIX', 18)

        # Get time to expiry
        time_to_expiry = self._get_time_to_expiry(timestamp)

        # Use realized volatility
        rv_20d = market_row.get('RV_20d', vix / 100)
        atm_iv = max(rv_20d, vix / 100)

        # Get current option prices
        ce_price = self.option_simulator.get_option_price(
            strike=self.current_position.ce_strike,
            spot=spot,
            atm_iv=atm_iv,
            time_to_expiry=max(time_to_expiry, 0.001),
            option_type='CE',
            price_type='ltp'
        )

        pe_price = self.option_simulator.get_option_price(
            strike=self.current_position.pe_strike,
            spot=spot,
            atm_iv=atm_iv,
            time_to_expiry=max(time_to_expiry, 0.001),
            option_type='PE',
            price_type='ltp'
        )

        # Calculate current P&L
        current_pnl = self.current_position.calculate_pnl(ce_price, pe_price)
        self.current_position.unrealized_pnl = current_pnl

        # Track max profit/loss
        self.current_position.max_profit = max(self.current_position.max_profit, current_pnl)
        self.current_position.max_loss = min(self.current_position.max_loss, current_pnl)

        # Calculate P&L percentage
        total_entry_premium = (
            self.current_position.ce_entry_price +
            self.current_position.pe_entry_price
        ) * self.current_position.lots * self.current_position.lot_size

        pnl_percent = current_pnl / total_entry_premium if total_entry_premium > 0 else 0

        # Check exit conditions
        exit_reason = None

        # 1. Profit target
        if pnl_percent >= self.strategy_config['profit_target']:
            exit_reason = "Profit target"

        # 2. Stop loss
        elif pnl_percent <= -self.strategy_config['stop_loss']:
            exit_reason = "Stop loss"

        # 3. Expiry day exit
        elif self._is_expiry_day(timestamp) and self.strategy_config['exit_on_expiry_day']:
            exit_time_str = self.strategy_config['expiry_exit_time']
            exit_hour, exit_minute = map(int, exit_time_str.split(':'))
            if timestamp.time() >= time(exit_hour, exit_minute):
                exit_reason = "Expiry day exit"

        # 4. Delta threshold
        elif self.strategy_config.get('delta_threshold', None):
            # Get current deltas
            ce_delta = self.bs_model.delta_call(
                spot, self.current_position.ce_strike,
                max(time_to_expiry, 0.001), atm_iv
            )
            pe_delta = self.bs_model.delta_put(
                spot, self.current_position.pe_strike,
                max(time_to_expiry, 0.001), atm_iv
            )

            portfolio_delta = self.current_position.calculate_portfolio_delta(ce_delta, pe_delta)

            if abs(portfolio_delta) > self.strategy_config['delta_threshold']:
                exit_reason = f"Delta breach ({portfolio_delta:.1f})"

        # Exit if any condition met
        if exit_reason:
            self._exit_position(timestamp, market_row, exit_reason)

    def _exit_position(
        self,
        timestamp: pd.Timestamp,
        market_row: pd.Series,
        exit_reason: str
    ) -> None:
        """Exit the current position"""
        if self.current_position is None:
            return

        spot = market_row['Close']
        vix = market_row.get('VIX', 18)

        # Get time to expiry
        time_to_expiry = self._get_time_to_expiry(timestamp)

        # Use realized volatility
        rv_20d = market_row.get('RV_20d', vix / 100)
        atm_iv = max(rv_20d, vix / 100)

        # Determine exit price based on position type
        position_type = self.current_position.position_type

        if position_type == PositionType.SHORT_STRANGLE:
            # Buying back - use ask price
            ce_exit_price = self.option_simulator.get_option_price(
                self.current_position.ce_strike, spot, atm_iv,
                max(time_to_expiry, 0.001), 'CE', 'ask'
            )
            pe_exit_price = self.option_simulator.get_option_price(
                self.current_position.pe_strike, spot, atm_iv,
                max(time_to_expiry, 0.001), 'PE', 'ask'
            )

            # Add slippage
            slippage = self.cost_config['stop_loss_slippage'] if 'stop' in exit_reason.lower() \
                else self.cost_config['exit_slippage']
            ce_exit_price += slippage
            pe_exit_price += slippage

        else:  # LONG_STRANGLE
            # Selling - use bid price
            ce_exit_price = self.option_simulator.get_option_price(
                self.current_position.ce_strike, spot, atm_iv,
                max(time_to_expiry, 0.001), 'CE', 'bid'
            )
            pe_exit_price = self.option_simulator.get_option_price(
                self.current_position.pe_strike, spot, atm_iv,
                max(time_to_expiry, 0.001), 'PE', 'bid'
            )

            # Add slippage
            slippage = self.cost_config['stop_loss_slippage'] if 'stop' in exit_reason.lower() \
                else self.cost_config['exit_slippage']
            ce_exit_price -= slippage
            pe_exit_price -= slippage

        # Calculate exit costs
        total_premium = (ce_exit_price + pe_exit_price)
        lots = self.current_position.lots
        exit_costs_detail = get_cost_per_trade(total_premium, lots)
        exit_costs = exit_costs_detail['total_cost']

        self.current_position.ce_exit_price = ce_exit_price
        self.current_position.pe_exit_price = pe_exit_price
        self.current_position.exit_time = timestamp
        self.current_position.exit_reason = exit_reason
        self.current_position.exit_costs = exit_costs
        self.current_position.status = PositionStatus.CLOSED

        # Calculate final P&L
        final_pnl = self.current_position.calculate_pnl(ce_exit_price, pe_exit_price)
        self.current_position.realized_pnl = final_pnl

        # Update capital
        self.current_capital += final_pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)

        # Calculate metrics
        total_entry_premium = (
            self.current_position.ce_entry_price +
            self.current_position.pe_entry_price
        ) * lots * self.current_position.lot_size

        pnl_percent = final_pnl / total_entry_premium if total_entry_premium > 0 else 0

        hold_time = (timestamp - self.current_position.entry_time).total_seconds() / 60

        # Create trade record
        trade = Trade(
            entry_time=self.current_position.entry_time,
            exit_time=timestamp,
            position_type=position_type.value,
            ce_strike=self.current_position.ce_strike,
            pe_strike=self.current_position.pe_strike,
            ce_entry_price=self.current_position.ce_entry_price,
            pe_entry_price=self.current_position.pe_entry_price,
            ce_exit_price=ce_exit_price,
            pe_exit_price=pe_exit_price,
            lots=lots,
            pnl=final_pnl,
            pnl_percent=pnl_percent,
            exit_reason=exit_reason,
            entry_vix=vix,  # Approximate
            exit_vix=vix,
            max_profit=self.current_position.max_profit,
            max_loss=self.current_position.max_loss,
            hold_time_minutes=int(hold_time),
            transaction_costs=self.current_position.entry_costs + exit_costs
        )

        self.trades.append(trade)

        logger.info(f"\n{'='*60}")
        logger.info(f"EXIT: {timestamp}")
        logger.info(f"  Reason: {exit_reason}")
        logger.info(f"  Spot: ₹{spot:,.2f} | VIX: {vix:.2f}")
        logger.info(f"  CE Exit: {self.current_position.ce_strike} @ ₹{ce_exit_price:.2f}")
        logger.info(f"  PE Exit: {self.current_position.pe_strike} @ ₹{pe_exit_price:.2f}")
        logger.info(f"  P&L: ₹{final_pnl:,.2f} ({pnl_percent*100:.2f}%)")
        logger.info(f"  Capital: ₹{self.current_capital:,.2f}")
        logger.info(f"  Hold time: {hold_time:.0f} minutes")
        logger.info(f"{'='*60}\n")

        # Clear position
        self.current_position = None

    def _select_strikes(
        self,
        chain: pd.DataFrame,
        spot: float
    ) -> Optional[Dict]:
        """Select strikes based on configured method"""
        method = self.strategy_config['strike_selection_method']

        if method == 'DELTA':
            # Select based on delta
            ce_target = abs(self.strategy_config['ce_delta_target'])
            pe_target = abs(self.strategy_config['pe_delta_target'])

            # Find CE strike
            chain['ce_delta_diff'] = abs(abs(chain['ce_delta']) - ce_target)
            ce_row = chain.loc[chain['ce_delta_diff'].idxmin()]

            # Find PE strike
            chain['pe_delta_diff'] = abs(abs(chain['pe_delta']) - pe_target)
            pe_row = chain.loc[chain['pe_delta_diff'].idxmin()]

            return {
                'CE': {'strike': ce_row['strike']},
                'PE': {'strike': pe_row['strike']}
            }

        elif method == 'ATM_OFFSET':
            # Select based on ATM offset
            offset = self.strategy_config['atm_offset']
            atm_strike = round(spot / 100) * 100

            return {
                'CE': {'strike': atm_strike + offset},
                'PE': {'strike': atm_strike - offset}
            }

        else:
            return None

    def _get_time_to_expiry(self, timestamp: pd.Timestamp) -> float:
        """Get time to next expiry in years"""
        # Find next expiry date
        future_expiries = [exp for exp in self.expiry_dates if exp > timestamp]

        if not future_expiries:
            return 0

        next_expiry = future_expiries[0]

        # Calculate time difference in days
        days_to_expiry = (next_expiry - timestamp).total_seconds() / (24 * 3600)

        # Convert to years
        time_to_expiry = days_to_expiry / 365.0

        return max(time_to_expiry, 0.001)  # Minimum 0.001 years

    def _is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """Check if current time is within trading hours"""
        # Skip first and last N minutes
        current_time = timestamp.time()

        # Market hours: 9:15 AM to 3:30 PM
        market_open = time(9, 15)
        market_close = time(15, 30)

        if current_time < market_open or current_time > market_close:
            return False

        # Skip first N minutes
        no_trade_before_minutes = self.strategy_config['no_trade_first_minutes']
        no_trade_start = time(9, 15 + no_trade_before_minutes)

        if current_time < no_trade_start:
            return False

        # Skip last N minutes
        no_trade_last_minutes = self.strategy_config['no_trade_last_minutes']
        no_trade_end_hour = 15
        no_trade_end_minute = 30 - no_trade_last_minutes

        no_trade_end = time(no_trade_end_hour, no_trade_end_minute)

        if current_time > no_trade_end:
            return False

        return True

    def _is_expiry_day(self, timestamp: pd.Timestamp) -> bool:
        """Check if current day is an expiry day"""
        date_only = timestamp.date()
        return any(exp.date() == date_only for exp in self.expiry_dates)

    def _is_major_event_day(self, timestamp: pd.Timestamp) -> bool:
        """Check if current day has major events"""
        if not EVENTS_CONFIG['skip_major_events']:
            return False

        date_str = timestamp.strftime('%Y-%m-%d')
        return date_str in EVENTS_CONFIG['major_events']

    def _calculate_iv_percentile(self, market_row: pd.Series) -> float:
        """Calculate IV percentile"""
        vix = market_row.get('VIX', 18)
        current_iv = vix / 100

        # Store in history
        self.iv_history.append(current_iv)

        # Keep only last 60 values
        if len(self.iv_history) > 60:
            self.iv_history = self.iv_history[-60:]

        if len(self.iv_history) < 10:
            return 50.0  # Not enough data

        # Calculate percentile
        percentile = (sum(1 for x in self.iv_history if x < current_iv) / len(self.iv_history)) * 100

        return percentile

    def _update_equity_curve(self, timestamp: pd.Timestamp, market_row: pd.Series) -> None:
        """Update equity curve"""
        # Calculate current equity
        current_equity = self.current_capital

        if self.current_position is not None:
            # Add unrealized P&L
            spot = market_row['Close']
            vix = market_row.get('VIX', 18)
            rv_20d = market_row.get('RV_20d', vix / 100)
            atm_iv = max(rv_20d, vix / 100)
            time_to_expiry = self._get_time_to_expiry(timestamp)

            ce_price = self.option_simulator.get_option_price(
                self.current_position.ce_strike, spot, atm_iv,
                max(time_to_expiry, 0.001), 'CE', 'ltp'
            )
            pe_price = self.option_simulator.get_option_price(
                self.current_position.pe_strike, spot, atm_iv,
                max(time_to_expiry, 0.001), 'PE', 'ltp'
            )

            unrealized_pnl = self.current_position.calculate_pnl(ce_price, pe_price)
            current_equity += unrealized_pnl

        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'returns': (current_equity / self.initial_capital - 1) * 100,
            'drawdown': (current_equity / self.peak_capital - 1) * 100 if self.peak_capital > 0 else 0
        })

    def _compile_results(self) -> Dict:
        """Compile backtest results"""
        results = {
            'trades': self.trades,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital - 1) * 100,
            'num_trades': len(self.trades),
            'strategy_config': self.strategy_config,
            'cost_config': self.cost_config,
        }

        return results

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'position_type': trade.position_type,
                'ce_strike': trade.ce_strike,
                'pe_strike': trade.pe_strike,
                'ce_entry_price': trade.ce_entry_price,
                'pe_entry_price': trade.pe_entry_price,
                'ce_exit_price': trade.ce_exit_price,
                'pe_exit_price': trade.pe_exit_price,
                'lots': trade.lots,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason,
                'hold_time_minutes': trade.hold_time_minutes,
                'transaction_costs': trade.transaction_costs,
            })

        return pd.DataFrame(trades_data)


if __name__ == "__main__":
    logger.info("Backtest engine module loaded")
