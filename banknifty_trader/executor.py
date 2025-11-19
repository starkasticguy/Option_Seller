"""
Main Execution Module
Real-time trading system with console display
"""

import time
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional
import pytz
from colorama import Fore, Style, init
import threading
import os

# Set TERM environment variable if not set (fixes terminal warnings)
if 'TERM' not in os.environ:
    os.environ['TERM'] = 'xterm-256color'

# Import Zerodha KiteConnect
try:
    from kiteconnect import KiteConnect
except ImportError:
    print("KiteConnect not installed. Install with: pip install kiteconnect")
    KiteConnect = None

# Import our modules
from config import (
    ZERODHA_CONFIG, TRADING_PARAMS, DISPLAY_CONFIG,
    INSTRUMENT_CONFIG, get_config_summary
)
from market_monitor import MarketMonitor
from strategy import BankNiftyOptionsTrader
from risk_manager import RiskManager
from database import TradingDatabase

# Initialize colorama
init(autoreset=True)


class TradingSystem:
    """Main trading system orchestrator"""

    def __init__(self, paper_trading: bool = True):
        """
        Initialize the trading system

        Args:
            paper_trading: If True, run in paper trading mode (no real orders)
        """
        self.paper_trading = paper_trading
        self.ist = pytz.timezone('Asia/Kolkata')
        self.running = False

        # Initialize components
        print(f"{Fore.CYAN}Initializing BankNifty Options Trading System...")

        # Display trading mode
        if paper_trading:
            print(f"{Fore.YELLOW}Running in PAPER TRADING mode (no real orders)")
        else:
            print(f"{Fore.RED}Running in LIVE TRADING mode (real orders will be placed)")

        # Zerodha connection - Initialize even in paper trading to allow authentication testing
        # Orders will only be blocked in paper trading mode, but you can still login and get market data
        self.kite = self.initialize_zerodha()

        # Initialize modules
        self.market_monitor = MarketMonitor(self.kite)
        self.trader = BankNiftyOptionsTrader(self.kite, self.market_monitor)
        self.risk_manager = RiskManager()
        self.database = TradingDatabase()

        # State tracking
        self.current_position = None
        self.position_start_time = None
        self.max_profit_seen = 0
        self.max_loss_seen = 0

        # Display settings
        self.update_interval = DISPLAY_CONFIG['update_interval']
        self.use_colors = DISPLAY_CONFIG['use_colors']
        self.clear_screen = DISPLAY_CONFIG['clear_screen']

        # Session tracking
        self.session_start = datetime.now(self.ist)
        self.trades_today = 0

    def initialize_zerodha(self) -> Optional[KiteConnect]:
        """
        Initialize Zerodha KiteConnect API

        Returns:
            KiteConnect instance or None
        """
        if not KiteConnect:
            print(f"{Fore.RED}KiteConnect not available. Running in mock mode.")
            return None

        try:
            print(f"{Fore.YELLOW}Initializing Zerodha connection...")

            api_key = ZERODHA_CONFIG['api_key']
            if api_key == 'YOUR_API_KEY':
                print(f"{Fore.RED}Please configure your Zerodha API credentials in config.py")
                print(f"{Fore.YELLOW}Continuing in paper trading mode...")
                return None

            kite = KiteConnect(api_key=api_key)

            # Check if we have a stored access token
            access_token = ZERODHA_CONFIG.get('access_token')

            if not access_token:
                # Need to login - Manual OAuth flow
                print(f"\n{Fore.CYAN}{'='*60}")
                print(f"{Fore.YELLOW}ZERODHA LOGIN REQUIRED - MANUAL AUTHENTICATION")
                print(f"{Fore.CYAN}{'='*60}\n")

                print(f"{Fore.WHITE}Step 1: Copy this URL and open it in your browser:")
                print(f"{Fore.CYAN}{kite.login_url()}\n")

                print(f"{Fore.WHITE}Step 2: Log in with your Zerodha credentials")
                print(f"{Fore.WHITE}        - Enter User ID and Password")
                print(f"{Fore.WHITE}        - Enter 2FA/TOTP if enabled\n")

                print(f"{Fore.WHITE}Step 3: After successful login, you'll be redirected to:")
                print(f"{Fore.WHITE}        127.0.0.1/?request_token=XXXXXXXXX&action=login&status=success\n")

                print(f"{Fore.WHITE}Step 4: Copy the 'request_token' value from the URL")
                print(f"{Fore.WHITE}        (everything after 'request_token=' and before '&')\n")

                print(f"{Fore.YELLOW}Paste the request token here and press Enter:")
                print(f"{Fore.CYAN}>>> {Fore.WHITE}", end='')

                request_token = input().strip()

                if not request_token:
                    print(f"{Fore.RED}No token provided. Exiting...")
                    return None

                print(f"\n{Fore.YELLOW}Generating session with request token...")

                data = kite.generate_session(
                    request_token,
                    api_secret=ZERODHA_CONFIG['api_secret']
                )
                access_token = data['access_token']

                print(f"{Fore.GREEN}✓ Login successful!")
                print(f"\n{Fore.YELLOW}{'='*60}")
                print(f"{Fore.YELLOW}IMPORTANT: Save this access token for future use")
                print(f"{Fore.YELLOW}{'='*60}")
                print(f"{Fore.WHITE}Access Token: {Fore.CYAN}{access_token}")
                print(f"\n{Fore.WHITE}Add this to your .env file:")
                print(f"{Fore.CYAN}ZERODHA_ACCESS_TOKEN={access_token}")
                print(f"{Fore.YELLOW}{'='*60}\n")

            kite.set_access_token(access_token)

            # Verify connection
            profile = kite.profile()
            print(f"{Fore.GREEN}Connected as: {profile['user_name']}")

            return kite

        except Exception as e:
            print(f"{Fore.RED}Error initializing Zerodha: {str(e)}")
            print(f"{Fore.YELLOW}Continuing in paper trading mode...")
            return None

    def clear_console(self):
        """Clear the console screen"""
        if self.clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')

    def display_header(self):
        """Display system header"""
        mode = "PAPER TRADING" if self.paper_trading else "LIVE TRADING"
        mode_color = Fore.YELLOW if self.paper_trading else Fore.RED

        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}    BANKNIFTY OPTIONS TRADING SYSTEM - {mode_color}{mode}")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.WHITE}Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.WHITE}Current Time: {datetime.now(self.ist).strftime('%H:%M:%S')}")

    def display_market_status(self):
        """Display current market conditions"""
        market_status = self.market_monitor.get_market_status()
        vix_data = market_status['vix']
        event_data = market_status['events']

        print(f"\n{Fore.YELLOW}📊 MARKET STATUS:")

        # BankNifty spot
        spot = self.trader.get_banknifty_spot()
        print(f"{Fore.WHITE}BankNifty: {Fore.CYAN}{spot:,.2f}")

        # VIX
        vix_color = vix_data['color']
        vix_symbol = self._get_vix_symbol(vix_data['current'])
        print(f"{Fore.WHITE}VIX: {vix_color}{vix_data['current']:.2f} ({vix_data['status']}) {vix_symbol}")

        # Event Risk
        event_symbol = self._get_event_symbol(event_data['status'])
        event_color = {
            'RED': Fore.RED,
            'YELLOW': Fore.YELLOW,
            'GREEN': Fore.GREEN
        }.get(event_data['status'], Fore.WHITE)

        print(f"{Fore.WHITE}Event Risk: {event_color}{event_data['status']} {event_symbol}")
        print(f"{Fore.WHITE}Message: {event_data['message']}")

        # Trading Status
        can_trade = market_status['can_trade']
        trade_symbol = '✅' if can_trade else '🛑'
        trade_color = Fore.GREEN if can_trade else Fore.RED
        print(f"{Fore.WHITE}Trading: {trade_color}{trade_symbol} {'ALLOWED' if can_trade else 'BLOCKED'}")

    def display_position_status(self):
        """Display current position and P&L"""
        print(f"\n{Fore.YELLOW}📈 CURRENT POSITION:")

        if self.current_position:
            # Calculate current P&L
            pnl = self.trader.calculate_pnl()
            pnl_pct = self.trader.calculate_pnl_percent()
            pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
            pnl_symbol = '📈' if pnl >= 0 else '📉'

            # Update max profit/loss
            if pnl > self.max_profit_seen:
                self.max_profit_seen = pnl
            if pnl < self.max_loss_seen:
                self.max_loss_seen = pnl

            # Position details
            print(f"{Fore.WHITE}Type: {Fore.CYAN}{self.current_position['type']}")
            print(f"{Fore.WHITE}Strikes: CE {self.current_position['ce_strike']}, PE {self.current_position['pe_strike']}")
            print(f"{Fore.WHITE}Entry: CE ₹{self.current_position['ce_entry_premium']:.2f}, PE ₹{self.current_position['pe_entry_premium']:.2f}")
            print(f"{Fore.WHITE}Entry Time: {self.position_start_time.strftime('%H:%M:%S')}")

            # Duration
            duration = datetime.now(self.ist) - self.position_start_time
            duration_mins = int(duration.total_seconds() / 60)
            print(f"{Fore.WHITE}Duration: {duration_mins} minutes")

            # P&L
            print(f"\n{Fore.WHITE}P&L: {pnl_color}{pnl_symbol} ₹{pnl:,.2f} ({pnl_pct:+.1%})")
            print(f"{Fore.WHITE}Peak Profit: {Fore.GREEN}₹{self.max_profit_seen:,.2f}")
            print(f"{Fore.WHITE}Max Loss: {Fore.RED}₹{self.max_loss_seen:,.2f}")

            # Targets
            lot_size = INSTRUMENT_CONFIG['lot_size']
            total_premium = (self.current_position['ce_entry_premium'] +
                           self.current_position['pe_entry_premium']) * lot_size
            profit_target = total_premium * TRADING_PARAMS['profit_target']
            stop_loss = total_premium * TRADING_PARAMS['stop_loss']

            print(f"{Fore.WHITE}Target: {Fore.GREEN}₹{profit_target:,.2f}")
            print(f"{Fore.WHITE}Stop: {Fore.RED}₹{-stop_loss:,.2f}")

            # Delta
            delta = self.trader.calculate_portfolio_delta()
            delta_color = Fore.GREEN if abs(delta) < TRADING_PARAMS['delta_threshold'] else Fore.YELLOW
            print(f"{Fore.WHITE}Delta: {delta_color}{delta:+.2f}")

        else:
            print(f"{Fore.WHITE}Status: {Fore.YELLOW}No open positions")
            print(f"{Fore.WHITE}Waiting for entry signal...")

    def display_signal(self):
        """Display current trading signal"""
        signal = self.trader.calculate_signals()

        print(f"\n{Fore.YELLOW}🎯 TRADING SIGNAL:")

        action = signal.get('action', 'SKIP')

        if action == 'BUY_STRANGLE':
            print(f"{Fore.GREEN}✅ ACTION: BUY STRANGLE")
            strikes = signal.get('strikes', {})
            if strikes:
                print(f"   {Fore.WHITE}Buy CE: {strikes['CE']['strike']} @ ₹{strikes['CE']['premium']:.2f}")
                print(f"   {Fore.WHITE}Buy PE: {strikes['PE']['strike']} @ ₹{strikes['PE']['premium']:.2f}")
            print(f"   {Fore.WHITE}Lots: {signal.get('lots', 1)}")

        elif action == 'SELL_STRANGLE':
            print(f"{Fore.GREEN}✅ ACTION: SELL STRANGLE")
            strikes = signal.get('strikes', {})
            if strikes:
                print(f"   {Fore.WHITE}Sell CE: {strikes['CE']['strike']} @ ₹{strikes['CE']['premium']:.2f}")
                print(f"   {Fore.WHITE}Sell PE: {strikes['PE']['strike']} @ ₹{strikes['PE']['premium']:.2f}")
            print(f"   {Fore.WHITE}Lots: {signal.get('lots', 1)}")

            if signal.get('hedge'):
                print(f"   {Fore.YELLOW}Hedge: {signal['hedge']}")

        elif action == 'HOLD':
            print(f"{Fore.BLUE}⏸️  ACTION: HOLD POSITION")
            pnl = signal.get('pnl', 0)
            pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
            print(f"   {Fore.WHITE}Current P&L: {pnl_color}₹{pnl:,.2f}")
            print(f"   {Fore.WHITE}Delta: {signal.get('delta', 0):+.2f}")

        elif action == 'EXIT':
            print(f"{Fore.RED}🛑 ACTION: EXIT ALL POSITIONS")
            pnl = signal.get('pnl', 0)
            pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
            print(f"   {Fore.WHITE}Final P&L: {pnl_color}₹{pnl:,.2f}")

        elif action == 'HEDGE':
            print(f"{Fore.YELLOW}⚖️  ACTION: HEDGE DELTA")
            print(f"   {Fore.WHITE}Direction: {signal.get('hedge_direction', 'UNKNOWN')}")
            print(f"   {Fore.WHITE}Quantity: {signal.get('hedge_quantity', 0):.2f} lots")

        else:  # SKIP
            print(f"{Fore.YELLOW}⏭️  ACTION: SKIP - WAIT")

        # Reason and confidence
        print(f"   {Fore.WHITE}Reason: {signal.get('reason', 'N/A')}")
        confidence = signal.get('confidence', 0)
        conf_color = Fore.GREEN if confidence > 0.7 else (Fore.YELLOW if confidence > 0.4 else Fore.RED)
        print(f"   {Fore.WHITE}Confidence: {conf_color}{confidence*100:.1f}%")

    def display_risk_metrics(self):
        """Display risk management metrics"""
        risk_summary = self.risk_manager.get_risk_summary()

        print(f"\n{Fore.YELLOW}⚠️  RISK METRICS:")

        # Daily P&L
        daily_pnl = risk_summary['daily_pnl']
        pnl_color = Fore.GREEN if daily_pnl >= 0 else Fore.RED
        pnl_pct = risk_summary['daily_pnl_pct']

        print(f"{Fore.WHITE}Daily P&L: {pnl_color}₹{daily_pnl:,.2f} ({pnl_pct:+.2f}%)")
        print(f"{Fore.WHITE}Daily Limit: ₹{risk_summary['daily_limit']:,.0f}")

        limit_remaining = risk_summary['limit_remaining']
        limit_color = Fore.GREEN if limit_remaining > 3000 else (Fore.YELLOW if limit_remaining > 1000 else Fore.RED)
        print(f"{Fore.WHITE}Remaining: {limit_color}₹{limit_remaining:,.2f}")

        # VaR
        var_95 = risk_summary['var_95']
        print(f"{Fore.WHITE}VaR (95%): ₹{var_95:,.0f}")

        # Positions
        print(f"{Fore.WHITE}Positions: {risk_summary['open_positions']}/{risk_summary['max_positions']}")
        print(f"{Fore.WHITE}Trades Today: {risk_summary['trades_today']}")

        # Streaks
        if risk_summary['win_streak'] > 0:
            print(f"{Fore.GREEN}Win Streak: {risk_summary['win_streak']}")
        elif risk_summary['loss_streak'] > 0:
            print(f"{Fore.RED}Loss Streak: {risk_summary['loss_streak']}")

        # Cooldown
        if risk_summary['in_cooldown']:
            print(f"{Fore.RED}⚠️  IN COOLDOWN MODE")

    def display_upcoming_events(self):
        """Display upcoming economic events"""
        event_data = self.market_monitor.check_economic_events()

        print(f"\n{Fore.YELLOW}📅 UPCOMING EVENTS (Next 24hrs):")

        events = event_data.get('events', [])
        if events:
            for i, event in enumerate(events[:3]):  # Show top 3
                impact = event.get('impact', 0)
                impact_str = '🔴 HIGH' if impact >= 4 else '🟡 MED'
                hours = event.get('hours_until', 0)
                print(f"   {impact_str} | {event['name']} in {hours:.1f}h")
        else:
            print(f"   {Fore.GREEN}✅ No high impact events")

    def display_footer(self):
        """Display footer with refresh info"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.WHITE}Refreshing in {self.update_interval} seconds... Press Ctrl+C to stop")
        print(f"{Fore.CYAN}{'='*60}")

    def _get_vix_symbol(self, vix: float) -> str:
        """Get symbol based on VIX level"""
        if vix > 25:
            return '🔴'
        elif vix > 20:
            return '🟡'
        elif vix > 15:
            return '🟢'
        else:
            return '🔵'

    def _get_event_symbol(self, status: str) -> str:
        """Get symbol based on event status"""
        return {
            'RED': '🔴',
            'YELLOW': '🟡',
            'GREEN': '🟢'
        }.get(status, '⚪')

    def check_auto_exits(self):
        """Check for automatic exit conditions"""
        if not self.current_position:
            return

        signal = self.trader.calculate_signals()

        if signal['action'] == 'EXIT':
            self.exit_position(signal.get('reason', 'Auto exit'))

    def check_hedge_requirements(self):
        """Check if hedging is needed"""
        if not self.current_position:
            return

        if not TRADING_PARAMS['use_hedging']:
            return

        signal = self.trader.calculate_signals()

        if signal['action'] == 'HEDGE':
            print(f"\n{Fore.YELLOW}Hedging required: {signal.get('reason')}")
            # TODO: Implement hedging logic

    def enter_position(self, signal: Dict):
        """
        Enter a new position

        Args:
            signal: Trading signal dictionary
        """
        try:
            if self.current_position:
                print(f"{Fore.YELLOW}Already in a position, skipping entry")
                return

            # Check risk limits
            allowed, reason = self.risk_manager.check_position_limits()
            if not allowed:
                print(f"{Fore.RED}Position blocked: {reason}")
                return

            strikes = signal.get('strikes', {})
            if not strikes:
                print(f"{Fore.RED}No strikes available")
                return

            # Create position
            self.current_position = {
                'type': 'SHORT_STRANGLE' if signal['action'] == 'SELL_STRANGLE' else 'LONG_STRANGLE',
                'ce_strike': strikes['CE']['strike'],
                'pe_strike': strikes['PE']['strike'],
                'ce_entry_premium': strikes['CE']['premium'],
                'pe_entry_premium': strikes['PE']['premium'],
                'lots': signal.get('lots', 1),
                'entry_vix': signal.get('vix', 0),
                'entry_spot': self.trader.get_banknifty_spot()
            }

            self.position_start_time = datetime.now(self.ist)
            self.max_profit_seen = 0
            self.max_loss_seen = 0

            # Update trader's position
            self.trader.current_position = self.current_position
            self.trader.entry_time = self.position_start_time

            # Add to risk manager
            self.risk_manager.open_positions.append(self.current_position)

            # Log to database
            if DATABASE_CONFIG['log_signals']:
                self.database.log_signal(signal)

            print(f"\n{Fore.GREEN}{'='*60}")
            print(f"{Fore.GREEN}POSITION ENTERED")
            print(f"{Fore.WHITE}Type: {self.current_position['type']}")
            print(f"{Fore.WHITE}CE: {self.current_position['ce_strike']} @ ₹{self.current_position['ce_entry_premium']:.2f}")
            print(f"{Fore.WHITE}PE: {self.current_position['pe_strike']} @ ₹{self.current_position['pe_entry_premium']:.2f}")
            print(f"{Fore.GREEN}{'='*60}\n")

            self.trades_today += 1

        except Exception as e:
            print(f"{Fore.RED}Error entering position: {str(e)}")

    def exit_position(self, reason: str = "Manual exit"):
        """
        Exit current position

        Args:
            reason: Exit reason
        """
        try:
            if not self.current_position:
                print(f"{Fore.YELLOW}No position to exit")
                return

            # Calculate final P&L
            final_pnl = self.trader.calculate_pnl()
            final_pnl_pct = self.trader.calculate_pnl_percent()

            # Get current market data
            chain = self.trader.get_option_chain()
            ce_row = chain[chain['strike'] == self.current_position['ce_strike']].iloc[0]
            pe_row = chain[chain['strike'] == self.current_position['pe_strike']].iloc[0]

            # Create trade record
            trade_data = {
                'entry_time': self.position_start_time,
                'exit_time': datetime.now(self.ist),
                'trade_type': self.current_position['type'],
                'ce_strike': self.current_position['ce_strike'],
                'pe_strike': self.current_position['pe_strike'],
                'ce_entry_premium': self.current_position['ce_entry_premium'],
                'pe_entry_premium': self.current_position['pe_entry_premium'],
                'ce_exit_premium': ce_row['ce_ltp'],
                'pe_exit_premium': pe_row['pe_ltp'],
                'lots': self.current_position['lots'],
                'lot_size': INSTRUMENT_CONFIG['lot_size'],
                'entry_spot': self.current_position['entry_spot'],
                'exit_spot': self.trader.get_banknifty_spot(),
                'entry_vix': self.current_position['entry_vix'],
                'exit_vix': self.market_monitor.get_india_vix()['current'],
                'pnl': final_pnl,
                'pnl_percent': final_pnl_pct,
                'exit_reason': reason,
                'entry_delta': 0,  # TODO: Store at entry
                'exit_delta': self.trader.calculate_portfolio_delta(),
                'max_profit': self.max_profit_seen,
                'max_loss': self.max_loss_seen,
                'duration_minutes': int((datetime.now(self.ist) - self.position_start_time).total_seconds() / 60),
                'commission': 100,  # TODO: Calculate actual commission
                'net_pnl': final_pnl - 100
            }

            # Log trade
            self.database.log_trade(trade_data)

            # Update risk manager
            self.risk_manager.record_trade({
                'pnl': final_pnl,
                'type': self.current_position['type'],
                'entry_time': self.position_start_time,
                'exit_time': datetime.now(self.ist),
                'exit_reason': reason
            })

            # Display exit summary
            pnl_color = Fore.GREEN if final_pnl >= 0 else Fore.RED
            print(f"\n{Fore.CYAN}{'='*60}")
            print(f"{Fore.CYAN}POSITION EXITED")
            print(f"{Fore.WHITE}Reason: {reason}")
            print(f"{Fore.WHITE}Duration: {trade_data['duration_minutes']} minutes")
            print(f"{Fore.WHITE}P&L: {pnl_color}₹{final_pnl:,.2f} ({final_pnl_pct:+.1%})")
            print(f"{Fore.CYAN}{'='*60}\n")

            # Clear position
            self.current_position = None
            self.trader.current_position = None
            self.position_start_time = None
            self.max_profit_seen = 0
            self.max_loss_seen = 0

        except Exception as e:
            print(f"{Fore.RED}Error exiting position: {str(e)}")

    def main_loop(self):
        """Main execution loop"""
        self.running = True

        print(f"\n{Fore.GREEN}System started successfully!")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop the system\n")

        time.sleep(2)

        while self.running:
            try:
                # Clear and display
                if self.clear_screen:
                    self.clear_console()

                self.display_header()
                self.display_market_status()
                self.display_position_status()
                self.display_signal()
                self.display_risk_metrics()
                self.display_upcoming_events()
                self.display_footer()

                # Check for automatic actions
                self.check_auto_exits()
                self.check_hedge_requirements()

                # Execute new signals (if auto-trading enabled)
                if not self.current_position:
                    signal = self.trader.calculate_signals()
                    if signal['action'] in ['BUY_STRANGLE', 'SELL_STRANGLE']:
                        if signal.get('confidence', 0) > 0.7:  # High confidence only
                            # TODO: Auto-execute or require confirmation
                            pass

                # Sleep before next update
                time.sleep(self.update_interval)

            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Shutdown requested by user...")
                self.cleanup_and_exit()
                break

            except Exception as e:
                print(f"\n{Fore.RED}Error in main loop: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

    def cleanup_and_exit(self):
        """Cleanup and exit gracefully"""
        print(f"\n{Fore.CYAN}Cleaning up...")

        # Exit any open positions
        if self.current_position:
            print(f"{Fore.YELLOW}Exiting open position...")
            self.exit_position("System shutdown")

        # Display session summary
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}SESSION SUMMARY")
        print(f"{Fore.CYAN}{'='*60}")

        stats = self.risk_manager.get_performance_stats()
        risk = self.risk_manager.get_risk_summary()

        print(f"\n{Fore.WHITE}Session Duration: {datetime.now(self.ist) - self.session_start}")
        print(f"{Fore.WHITE}Trades: {self.trades_today}")
        print(f"{Fore.WHITE}Daily P&L: {Fore.GREEN if risk['daily_pnl'] >= 0 else Fore.RED}₹{risk['daily_pnl']:,.2f}")

        if stats['total_trades'] > 0:
            print(f"{Fore.WHITE}Win Rate: {stats['win_rate']:.1%}")
            print(f"{Fore.WHITE}Profit Factor: {stats['profit_factor']:.2f}")

        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.GREEN}System stopped successfully!")

        # Close database
        self.database.close()

        self.running = False

    def run(self):
        """Start the trading system"""
        # Display config
        print(get_config_summary())

        # Confirm start
        if not self.paper_trading:
            print(f"\n{Fore.RED}{'='*60}")
            print(f"{Fore.RED}WARNING: LIVE TRADING MODE")
            print(f"{Fore.RED}Real orders will be placed!")
            print(f"{Fore.RED}{'='*60}")
            confirm = input(f"\n{Fore.YELLOW}Type 'YES' to continue: ")
            if confirm != 'YES':
                print(f"{Fore.YELLOW}Cancelled by user")
                return

        # Start main loop
        self.main_loop()


def main():
    """Entry point"""
    import sys

    # Check for command line arguments
    paper_trading = True
    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        paper_trading = False

    # Create and run system
    system = TradingSystem(paper_trading=paper_trading)
    system.run()


if __name__ == "__main__":
    main()
