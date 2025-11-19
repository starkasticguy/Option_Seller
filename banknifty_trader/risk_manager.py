"""
Risk Management Module
Handles position sizing, risk limits, and portfolio management
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pytz
from colorama import Fore, Style

from config import TRADING_PARAMS, RISK_RULES, INSTRUMENT_CONFIG


class RiskManager:
    """Comprehensive risk management system"""

    def __init__(self, capital: float = None, max_risk: float = None):
        """
        Initialize Risk Manager

        Args:
            capital: Total trading capital
            max_risk: Maximum risk per trade (as decimal)
        """
        self.capital = capital or TRADING_PARAMS['capital']
        self.max_risk = max_risk or TRADING_PARAMS['max_risk_per_trade']
        self.ist = pytz.timezone('Asia/Kolkata')

        # Track positions and P&L
        self.open_positions = []
        self.closed_positions = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0

        # Risk metrics
        self.daily_loss_limit = RISK_RULES['daily_loss_limit']
        self.consecutive_losses = 0
        self.max_consecutive_losses = RISK_RULES['max_consecutive_losses']
        self.in_cooldown = False
        self.cooldown_until = None

        # Trade statistics
        self.trade_history = []
        self.win_streak = 0
        self.loss_streak = 0

    def check_position_limits(self, new_position: Dict = None) -> Tuple[bool, str]:
        """
        Check if new position is within limits

        Args:
            new_position: Position details dictionary

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check if in cooldown
        if self.in_cooldown:
            if datetime.now(self.ist) < self.cooldown_until:
                remaining = (self.cooldown_until - datetime.now(self.ist)).seconds // 60
                return False, f"In cooldown for {remaining} more minutes"
            else:
                self.in_cooldown = False
                self.consecutive_losses = 0

        # Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            return False, f"Daily loss limit reached (₹{abs(self.daily_pnl):,.0f})"

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._enter_cooldown()
            return False, f"Max consecutive losses ({self.max_consecutive_losses}) reached"

        # Check max positions
        max_positions = TRADING_PARAMS['max_positions']
        if len(self.open_positions) >= max_positions:
            return False, f"Max positions ({max_positions}) already open"

        # Check portfolio loss
        max_portfolio_loss = TRADING_PARAMS['max_portfolio_loss']
        portfolio_loss_pct = abs(self.total_pnl / self.capital) if self.total_pnl < 0 else 0

        if portfolio_loss_pct >= max_portfolio_loss:
            return False, f"Portfolio loss limit reached ({portfolio_loss_pct:.1%})"

        return True, "Position limits OK"

    def calculate_position_size(self, volatility: float = None,
                               market_conditions: Dict = None) -> int:
        """
        Calculate optimal position size based on risk parameters

        Args:
            volatility: Current market volatility (VIX)
            market_conditions: Market conditions dict from market monitor

        Returns:
            Number of lots to trade
        """
        # Base position size
        base_size = TRADING_PARAMS['position_size']

        # Adjust for volatility
        if volatility:
            if volatility > 22:
                base_size = int(base_size * 0.5)
            elif volatility > 20:
                base_size = int(base_size * 0.75)

        # Adjust for market conditions
        if market_conditions:
            event_status = market_conditions.get('events', {}).get('status', 'GREEN')
            if event_status == 'YELLOW':
                base_size = int(base_size * 0.5)
            elif event_status == 'RED':
                base_size = 0

        # Adjust based on recent performance
        if self.loss_streak >= 2:
            base_size = int(base_size * 0.5)
        elif self.win_streak >= 3:
            base_size = min(base_size + 1, TRADING_PARAMS['position_size'] * 2)

        # Kelly Criterion adjustment (optional)
        if len(self.trade_history) > 20:
            kelly_fraction = self._calculate_kelly_criterion()
            base_size = int(base_size * kelly_fraction)

        # Ensure minimum of 0 lots
        return max(0, base_size)

    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            confidence: Confidence level (default 95%)

        Returns:
            VaR amount in rupees
        """
        if len(self.trade_history) < 10:
            # Not enough data, use conservative estimate
            return self.capital * 0.02  # 2% of capital

        # Get daily P&L from history
        daily_pnl = [trade['pnl'] for trade in self.trade_history[-60:]]

        # Calculate VaR using historical method
        daily_pnl_sorted = sorted(daily_pnl)
        var_index = int(len(daily_pnl_sorted) * (1 - confidence))
        var_value = abs(daily_pnl_sorted[var_index])

        return var_value

    def calculate_expected_shortfall(self, confidence: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)

        Args:
            confidence: Confidence level

        Returns:
            Expected shortfall amount
        """
        if len(self.trade_history) < 10:
            return self.capital * 0.03

        daily_pnl = [trade['pnl'] for trade in self.trade_history[-60:]]
        var = self.calculate_var(confidence)

        # Calculate average of losses beyond VaR
        tail_losses = [pnl for pnl in daily_pnl if pnl <= -var]

        if tail_losses:
            return abs(np.mean(tail_losses))
        else:
            return var

    def emergency_exit_check(self, current_pnl: float,
                            market_data: Dict = None) -> Tuple[bool, str]:
        """
        Check if emergency exit is needed

        Args:
            current_pnl: Current P&L
            market_data: Current market data

        Returns:
            Tuple of (exit_needed: bool, reason: str)
        """
        reasons = []

        # Check daily loss
        if self.daily_pnl + current_pnl <= -self.daily_loss_limit:
            reasons.append(f"Daily loss limit (₹{abs(self.daily_pnl + current_pnl):,.0f})")

        # Check emergency exit loss
        emergency_loss_pct = RISK_RULES['emergency_exit_loss']
        if (self.daily_pnl + current_pnl) / self.capital <= -emergency_loss_pct:
            reasons.append(f"Emergency loss threshold ({emergency_loss_pct:.1%})")

        # Check VIX spike
        if market_data and market_data.get('vix'):
            vix_data = market_data['vix']
            if vix_data.get('change_percent', 0) > RISK_RULES['max_sudden_vix_spike'] * 100:
                reasons.append(f"VIX spike ({vix_data['change_percent']:.1f}%)")

        # Check underlying move
        if market_data and market_data.get('underlying_move'):
            move_pct = market_data['underlying_move']
            if abs(move_pct) > RISK_RULES['max_underlying_move']:
                reasons.append(f"Large underlying move ({move_pct:.1%})")

        if reasons:
            return True, "; ".join(reasons)

        return False, ""

    def validate_greeks(self, greeks: Dict) -> Tuple[bool, str]:
        """
        Validate portfolio greeks are within limits

        Args:
            greeks: Portfolio greeks dictionary

        Returns:
            Tuple of (valid: bool, reason: str)
        """
        # Check delta
        delta = greeks.get('delta', 0)
        if abs(delta) > TRADING_PARAMS['delta_threshold']:
            return False, f"Delta out of range ({delta:.1f})"

        # Check gamma
        gamma = greeks.get('gamma', 0)
        if abs(gamma) > RISK_RULES['max_portfolio_gamma']:
            return False, f"Gamma too high ({gamma:.0f})"

        # Check vega
        vega = greeks.get('vega', 0)
        if abs(vega) > RISK_RULES['max_portfolio_vega']:
            return False, f"Vega too high ({vega:.0f})"

        # Check theta
        theta = greeks.get('theta', 0)
        if theta < RISK_RULES['max_portfolio_theta']:
            return False, f"Theta too negative ({theta:.0f})"

        return True, "Greeks within limits"

    def update_daily_pnl(self, pnl: float):
        """
        Update daily P&L

        Args:
            pnl: P&L to add
        """
        self.daily_pnl += pnl
        self.total_pnl += pnl

    def record_trade(self, trade: Dict):
        """
        Record a completed trade

        Args:
            trade: Trade details dictionary
        """
        self.trade_history.append({
            'timestamp': datetime.now(self.ist),
            'pnl': trade.get('pnl', 0),
            'type': trade.get('type', ''),
            'strikes': trade.get('strikes', {}),
            'lots': trade.get('lots', 1),
            'entry_time': trade.get('entry_time'),
            'exit_time': trade.get('exit_time'),
            'exit_reason': trade.get('exit_reason', '')
        })

        # Update P&L
        pnl = trade.get('pnl', 0)
        self.update_daily_pnl(pnl)

        # Update win/loss streaks
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            self.consecutive_losses = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.consecutive_losses += 1

        # Move from open to closed positions
        if trade in self.open_positions:
            self.open_positions.remove(trade)
        self.closed_positions.append(trade)

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of each trading day)"""
        self.daily_pnl = 0.0
        # Don't reset consecutive losses (carries over days)

    def _enter_cooldown(self):
        """Enter cooldown period after max consecutive losses"""
        self.in_cooldown = True
        cooldown_minutes = RISK_RULES['cooldown_period_minutes']
        self.cooldown_until = datetime.now(self.ist) + timedelta(minutes=cooldown_minutes)
        print(f"{Fore.RED}Entering cooldown for {cooldown_minutes} minutes")

    def _calculate_kelly_criterion(self) -> float:
        """
        Calculate Kelly Criterion for position sizing

        Returns:
            Kelly fraction (0-1)
        """
        if len(self.trade_history) < 20:
            return 1.0

        # Get win rate and average win/loss
        wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]

        if not wins or not losses:
            return 1.0

        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 1.0

        # Kelly formula: (W * R - L) / R
        # W = win rate, L = loss rate, R = win/loss ratio
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Use fractional Kelly (more conservative)
        kelly_fraction = max(0.0, min(kelly * 0.25, 1.0))  # 1/4 Kelly

        return kelly_fraction

    def get_risk_summary(self) -> Dict:
        """
        Get comprehensive risk summary

        Returns:
            Risk summary dictionary
        """
        return {
            'capital': self.capital,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.capital) * 100,
            'total_pnl': self.total_pnl,
            'daily_limit': self.daily_loss_limit,
            'limit_remaining': self.daily_loss_limit + self.daily_pnl,
            'var_95': self.calculate_var(0.95),
            'expected_shortfall': self.calculate_expected_shortfall(0.95),
            'open_positions': len(self.open_positions),
            'max_positions': TRADING_PARAMS['max_positions'],
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'consecutive_losses': self.consecutive_losses,
            'in_cooldown': self.in_cooldown,
            'trades_today': len([t for t in self.trade_history
                               if t['timestamp'].date() == datetime.now(self.ist).date()])
        }

    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics

        Returns:
            Performance statistics dictionary
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

        trades = self.trade_history
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Sharpe ratio (simplified)
        pnl_values = [t['pnl'] for t in trades]
        if len(pnl_values) > 1:
            sharpe_ratio = np.mean(pnl_values) / np.std(pnl_values) if np.std(pnl_values) > 0 else 0
        else:
            sharpe_ratio = 0

        # Max drawdown
        cumulative_pnl = np.cumsum([t['pnl'] for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        return {
            'total_trades': total_trades,
            'winners': len(wins),
            'losers': len(losses),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_pnl': self.total_pnl
        }

    def display_risk_summary(self):
        """Display formatted risk summary"""
        summary = self.get_risk_summary()
        stats = self.get_performance_stats()

        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}RISK MANAGEMENT SUMMARY")
        print(f"{Fore.CYAN}{'='*60}")

        # Daily P&L
        pnl_color = Fore.GREEN if summary['daily_pnl'] >= 0 else Fore.RED
        print(f"\n{Fore.YELLOW}Daily P&L: {pnl_color}₹{summary['daily_pnl']:,.2f} ({summary['daily_pnl_pct']:+.2f}%)")
        print(f"{Fore.WHITE}Daily Limit: ₹{summary['daily_limit']:,.0f}")
        print(f"{Fore.WHITE}Remaining: ₹{summary['limit_remaining']:,.2f}")

        # Risk Metrics
        print(f"\n{Fore.YELLOW}Risk Metrics:")
        print(f"{Fore.WHITE}VaR (95%): ₹{summary['var_95']:,.0f}")
        print(f"{Fore.WHITE}Expected Shortfall: ₹{summary['expected_shortfall']:,.0f}")

        # Positions
        print(f"\n{Fore.YELLOW}Positions:")
        print(f"{Fore.WHITE}Open: {summary['open_positions']}/{summary['max_positions']}")
        print(f"{Fore.WHITE}Trades Today: {summary['trades_today']}")

        # Streaks
        if summary['win_streak'] > 0:
            print(f"{Fore.GREEN}Win Streak: {summary['win_streak']}")
        elif summary['loss_streak'] > 0:
            print(f"{Fore.RED}Loss Streak: {summary['loss_streak']}")

        # Performance
        if stats['total_trades'] > 0:
            print(f"\n{Fore.YELLOW}Performance Stats:")
            print(f"{Fore.WHITE}Total Trades: {stats['total_trades']}")
            print(f"{Fore.WHITE}Win Rate: {stats['win_rate']:.1%}")
            print(f"{Fore.WHITE}Profit Factor: {stats['profit_factor']:.2f}")
            print(f"{Fore.WHITE}Sharpe Ratio: {stats['sharpe_ratio']:.2f}")

        print(f"\n{Fore.CYAN}{'='*60}\n")


if __name__ == "__main__":
    print("Testing Risk Manager...")
    rm = RiskManager(capital=300000)

    # Test position limits
    allowed, reason = rm.check_position_limits()
    print(f"Position Allowed: {allowed} - {reason}")

    # Test position sizing
    size = rm.calculate_position_size(volatility=18)
    print(f"Position Size: {size} lots")

    # Test VaR
    var = rm.calculate_var()
    print(f"VaR (95%): ₹{var:,.0f}")

    # Display summary
    rm.display_risk_summary()
