"""
Data Collection Module
Real-time market data collection system for BankNifty options
Collects option chain data, Greeks, volatility metrics for analysis
"""

import time
import sys
from datetime import datetime
from typing import Dict, Optional
import pytz
from colorama import Fore, Style, init
import os

# Set TERM environment variable if not set
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
    ZERODHA_CONFIG, DATA_COLLECTION, DISPLAY_CONFIG,
    INSTRUMENT_CONFIG, COLLECTION_HOURS, get_config_summary
)
from market_monitor import MarketMonitor
from data_analyzer import OptionsDataAnalyzer
from database import MarketDataDatabase

# Initialize colorama
init(autoreset=True)


class DataCollectionSystem:
    """Main data collection system orchestrator"""

    def __init__(self):
        """Initialize the data collection system"""
        self.ist = pytz.timezone('Asia/Kolkata')
        self.running = False

        # Initialize components
        print(f"{Fore.CYAN}Initializing BankNifty Options Data Collection System...")

        # Zerodha connection
        self.kite = self.initialize_zerodha()

        # Initialize modules
        self.market_monitor = MarketMonitor(self.kite)
        self.data_analyzer = OptionsDataAnalyzer(self.kite, self.market_monitor)
        self.database = MarketDataDatabase()

        # Display settings
        self.update_interval = DISPLAY_CONFIG['update_interval']
        self.use_colors = DISPLAY_CONFIG['use_colors']
        self.clear_screen = DISPLAY_CONFIG['clear_screen']

        # Session tracking
        self.session_start = datetime.now(self.ist)
        self.snapshots_collected = 0
        self.last_option_chain_snapshot = None
        self.last_volatility_calc = None

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
                print(f"{Fore.RED}Please configure your Zerodha API credentials in .env file")
                print(f"{Fore.YELLOW}Continuing in mock data mode...")
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
            print(f"{Fore.YELLOW}Continuing in mock data mode...")
            return None

    def clear_console(self):
        """Clear the console screen"""
        if self.clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')

    def display_header(self):
        """Display system header"""
        print(f"{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}    BANKNIFTY OPTIONS DATA COLLECTION SYSTEM")
        print(f"{Fore.CYAN}{'='*70}")
        print(f"{Fore.WHITE}Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.WHITE}Current Time: {datetime.now(self.ist).strftime('%H:%M:%S')}")
        print(f"{Fore.WHITE}Snapshots Collected: {self.snapshots_collected}")

    def display_market_status(self):
        """Display current market conditions"""
        print(f"\n{Fore.YELLOW}📊 MARKET DATA:")

        # BankNifty spot
        spot = self.data_analyzer.get_banknifty_spot()
        print(f"{Fore.WHITE}BankNifty Spot: {Fore.CYAN}{spot:,.2f}")

        # VIX
        vix_data = self.market_monitor.get_india_vix()
        vix_color = vix_data['color']
        vix_symbol = self._get_vix_symbol(vix_data['current'])
        print(f"{Fore.WHITE}India VIX: {vix_color}{vix_data['current']:.2f} ({vix_data['status']}) {vix_symbol}")
        print(f"{Fore.WHITE}  Change: {vix_data['change']:+.2f} ({vix_data['change_percent']:+.1f}%)")

        # Volatility
        volatility_data = self.market_monitor.calculate_banknifty_volatility()
        vol_color = volatility_data['color']
        print(f"{Fore.WHITE}Historical Volatility: {vol_color}{volatility_data['current']:.2f}% ({volatility_data['status']})")
        print(f"{Fore.WHITE}  Daily Vol: {volatility_data['daily']:.2f}% | Window: {volatility_data['period']} days")

        # Collection status
        is_market_open = self.is_market_hours()
        market_symbol = '✅' if is_market_open else '🛑'
        market_color = Fore.GREEN if is_market_open else Fore.RED
        print(f"{Fore.WHITE}Market Status: {market_color}{market_symbol} {'OPEN - COLLECTING DATA' if is_market_open else 'CLOSED'}")

    def display_option_chain_summary(self):
        """Display option chain summary"""
        print(f"\n{Fore.YELLOW}📈 OPTION CHAIN DATA:")

        try:
            chain = self.data_analyzer.get_option_chain()

            if chain.empty:
                print(f"{Fore.YELLOW}  No option chain data available")
                return

            # ATM data
            atm_row = chain[chain['is_atm']].iloc[0]
            atm_strike = atm_row['strike']

            print(f"{Fore.WHITE}ATM Strike: {Fore.CYAN}{atm_strike:,.0f}")
            print(f"{Fore.WHITE}CE Premium: ₹{atm_row['ce_ltp']:.2f} | PE Premium: ₹{atm_row['pe_ltp']:.2f}")
            print(f"{Fore.WHITE}CE IV: {atm_row['ce_iv']:.2f}% | PE IV: {atm_row['pe_iv']:.2f}%")

            # PCR
            total_ce_oi = chain['ce_oi'].sum()
            total_pe_oi = chain['pe_oi'].sum()
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            pcr_color = Fore.GREEN if 0.8 <= pcr <= 1.2 else Fore.YELLOW
            print(f"{Fore.WHITE}Put-Call Ratio (OI): {pcr_color}{pcr:.2f}")

            # Total OI
            print(f"{Fore.WHITE}Total CE OI: {total_ce_oi:,.0f} | Total PE OI: {total_pe_oi:,.0f}")

            # Last snapshot time
            if self.last_option_chain_snapshot:
                elapsed = (datetime.now(self.ist) - self.last_option_chain_snapshot).seconds
                print(f"{Fore.WHITE}Last Snapshot: {elapsed}s ago")

        except Exception as e:
            print(f"{Fore.RED}Error displaying option chain: {str(e)}")

    def display_greeks_summary(self):
        """Display Greeks summary"""
        if not DISPLAY_CONFIG['show_greeks']:
            return

        print(f"\n{Fore.YELLOW}📊 GREEKS ANALYSIS:")

        try:
            chain = self.data_analyzer.get_option_chain()

            if chain.empty:
                print(f"{Fore.YELLOW}  No greeks data available")
                return

            # ATM greeks
            atm_row = chain[chain['is_atm']].iloc[0]

            print(f"{Fore.WHITE}ATM CE Greeks:")
            print(f"  Delta: {atm_row['ce_delta']:.3f} | Gamma: {atm_row['ce_gamma']:.4f}")
            print(f"  Theta: {atm_row['ce_theta']:.2f} | Vega: {atm_row['ce_vega']:.2f}")

            print(f"{Fore.WHITE}ATM PE Greeks:")
            print(f"  Delta: {atm_row['pe_delta']:.3f} | Gamma: {atm_row['pe_gamma']:.4f}")
            print(f"  Theta: {atm_row['pe_theta']:.2f} | Vega: {atm_row['pe_vega']:.2f}")

        except Exception as e:
            print(f"{Fore.YELLOW}Greeks calculation unavailable: {str(e)}")

    def display_collection_stats(self):
        """Display data collection statistics"""
        print(f"\n{Fore.YELLOW}📁 COLLECTION STATISTICS:")

        duration = datetime.now(self.ist) - self.session_start
        duration_mins = int(duration.total_seconds() / 60)

        print(f"{Fore.WHITE}Session Duration: {duration_mins} minutes")
        print(f"{Fore.WHITE}Total Snapshots: {self.snapshots_collected}")

        if duration_mins > 0:
            rate = self.snapshots_collected / duration_mins
            print(f"{Fore.WHITE}Collection Rate: {rate:.1f} snapshots/min")

        # Database stats
        stats = self.database.get_collection_stats()
        if stats:
            print(f"{Fore.WHITE}DB Records:")
            print(f"  Market Data: {stats.get('market_data_count', 0):,}")
            print(f"  Option Chain: {stats.get('option_chain_count', 0):,}")
            print(f"  Volatility: {stats.get('volatility_count', 0):,}")

    def display_footer(self):
        """Display footer with refresh info"""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.WHITE}Refreshing in {self.update_interval} seconds... Press Ctrl+C to stop")
        print(f"{Fore.CYAN}{'='*70}")

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

    def is_market_hours(self) -> bool:
        """Check if currently in market hours"""
        now = datetime.now(self.ist)
        current_time = now.strftime('%H:%M')

        # Check if weekend
        if now.weekday() >= 5:
            return False

        # Check trading hours
        start_time = COLLECTION_HOURS['start_time']
        end_time = COLLECTION_HOURS['end_time']

        return start_time <= current_time <= end_time

    def collect_market_snapshot(self):
        """Collect and store market data snapshot"""
        try:
            # Get market data
            spot = self.data_analyzer.get_banknifty_spot()
            vix_data = self.market_monitor.get_india_vix()
            volatility_data = self.market_monitor.calculate_banknifty_volatility()

            # Store in database
            market_data = {
                'timestamp': datetime.now(self.ist),
                'spot': spot,
                'vix': vix_data['current'],
                'vix_change_pct': vix_data['change_percent'],
                'hv': volatility_data['current'],
                'hv_daily': volatility_data['daily']
            }

            self.database.log_market_data(market_data)
            self.snapshots_collected += 1

        except Exception as e:
            print(f"{Fore.RED}Error collecting market snapshot: {str(e)}")

    def collect_option_chain_snapshot(self):
        """Collect and store option chain snapshot"""
        try:
            chain = self.data_analyzer.get_option_chain()

            if chain.empty:
                return

            # Store in database
            self.database.log_option_chain(chain)
            self.last_option_chain_snapshot = datetime.now(self.ist)

        except Exception as e:
            print(f"{Fore.RED}Error collecting option chain: {str(e)}")

    def collect_volatility_data(self):
        """Collect and store volatility metrics"""
        try:
            vix_data = self.market_monitor.get_india_vix()
            hv_data = self.market_monitor.calculate_banknifty_volatility()

            # Calculate IV percentile
            iv_percentile = self.data_analyzer.calculate_iv_percentile()

            volatility_data = {
                'timestamp': datetime.now(self.ist),
                'iv': vix_data['current'],
                'hv': hv_data['current'],
                'iv_percentile': iv_percentile,
                'hv_window': hv_data['period']
            }

            self.database.log_volatility(volatility_data)
            self.last_volatility_calc = datetime.now(self.ist)

        except Exception as e:
            print(f"{Fore.RED}Error collecting volatility data: {str(e)}")

    def main_loop(self):
        """Main data collection loop"""
        self.running = True

        print(f"\n{Fore.GREEN}Data collection system started successfully!")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop the system\n")

        time.sleep(2)

        market_data_counter = 0
        option_chain_counter = 0
        volatility_counter = 0

        while self.running:
            try:
                # Clear and display
                if self.clear_screen:
                    self.clear_console()

                self.display_header()
                self.display_market_status()
                self.display_option_chain_summary()
                self.display_greeks_summary()
                self.display_collection_stats()
                self.display_footer()

                # Collect data at specified intervals
                market_data_counter += self.update_interval
                option_chain_counter += self.update_interval
                volatility_counter += self.update_interval

                # Market data snapshot (every 5 seconds)
                if market_data_counter >= DATA_COLLECTION['market_data_interval']:
                    self.collect_market_snapshot()
                    market_data_counter = 0

                # Option chain snapshot (every 30 seconds)
                if option_chain_counter >= DATA_COLLECTION['option_chain_interval']:
                    self.collect_option_chain_snapshot()
                    option_chain_counter = 0

                # Volatility data (every 60 seconds)
                if volatility_counter >= DATA_COLLECTION['volatility_calc_interval']:
                    self.collect_volatility_data()
                    volatility_counter = 0

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

        # Display session summary
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}SESSION SUMMARY")
        print(f"{Fore.CYAN}{'='*70}")

        duration = datetime.now(self.ist) - self.session_start
        duration_mins = int(duration.total_seconds() / 60)

        print(f"\n{Fore.WHITE}Session Duration: {duration_mins} minutes")
        print(f"{Fore.WHITE}Total Snapshots Collected: {self.snapshots_collected}")

        stats = self.database.get_collection_stats()
        if stats:
            print(f"\n{Fore.WHITE}Data Collected:")
            print(f"  Market Data Records: {stats.get('market_data_count', 0):,}")
            print(f"  Option Chain Records: {stats.get('option_chain_count', 0):,}")
            print(f"  Volatility Records: {stats.get('volatility_count', 0):,}")

        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.GREEN}Data collection stopped successfully!")
        print(f"{Fore.WHITE}Database: {self.database.db_path}")

        # Close database
        self.database.close()

        self.running = False

    def run(self):
        """Start the data collection system"""
        # Display config
        print(get_config_summary())

        # Confirm start
        print(f"\n{Fore.CYAN}Starting data collection...")
        time.sleep(1)

        # Start main loop
        self.main_loop()


def main():
    """Entry point"""
    # Create and run system
    system = DataCollectionSystem()
    system.run()


if __name__ == "__main__":
    main()
