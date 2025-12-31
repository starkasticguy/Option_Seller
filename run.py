#!/usr/bin/env python3
"""
Quick Start Script for BankNifty Data Collection System
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from banknifty_trader.data_collector import main

if __name__ == "__main__":
    main()
