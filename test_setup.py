#!/usr/bin/env python3
"""
Test script to validate bot setup and configuration
"""

import os
import sys
import json
from decimal import Decimal
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 50)
    print(f" {text}")
    print("=" * 50)

def check_status(condition, message):
    """Print status with emoji"""
    if condition:
        print(f"✅ {message}")
        return True
    else:
        print(f"❌ {message}")
        return False

def test_environment():
    """Test environment variables"""
    print_header("Testing Environment Variables")
    
    required_vars = [
        "DISCORD_BOT_TOKEN",
        "DISCORD_CHANNEL_ID", 
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET"
    ]
    
    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith("your_"):
            all_good = check_status(False, f"{var} is not properly configured")
        else:
            check_status(True, f"{var} is set")
    
    return all_good

def test_imports():
    """Test required imports"""
    print_header("Testing Required Libraries")
    
    libraries = [
        ("discord", "discord.py"),
        ("binance.um_futures", "binance-connector"),
        ("dotenv", "python-dotenv"),
        ("flask", "flask"),
        ("psutil", "psutil")
    ]
    
    all_good = True
    for module, package in libraries:
        try:
            __import__(module)
            check_status(True, f"{package} is installed")
        except ImportError:
            all_good = check_status(False, f"{package} is NOT installed - run: pip install {package}")
    
    return all_good

def test_signal_parsing():
    """Test signal parser"""
    print_header("Testing Signal Parser")
    
    # Import the parser
    try:
        from main import SignalParser, TradingSignal, OrderSide
        check_status(True, "Signal parser imported successfully")
    except ImportError as e:
        check_status(False, f"Failed to import parser: {e}")
        return False
    
    # Test signal
    test_message = """
    ATLAS-7 SIGNAL: BTCUSDT LONG
    Entry 1: 45000
    Entry 2: 44800
    Stop Loss: 44000
    TP 1: 46000 - 25%
    TP 2: 47000 - 35%
    TP 3: 48000 - 25%
    TP 4: 49000 - 15%
    """
    
    parser = SignalParser()
    signal = parser.parse(test_message, 123456)
    
    if signal:
        check_status(True, f"Signal parsed: {signal.symbol} {signal.side.name}")
        check_status(signal.validate(), "Signal validation passed")
        rr = signal.calculate_risk_reward()
        check_status(rr > 0, f"Risk-Reward calculated: {rr:.2f}")
        return True
    else:
        check_status(False, "Failed to parse test signal")
        return False

def test_binance_connection():
    """Test Binance API connection"""
    print_header("Testing Binance Connection")
    
    try:
        from binance.um_futures import UMFutures
        
        use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
        base_url = "https://testnet.binancefuture.com" if use_testnet else "https://fapi.binance.com"
        
        client = UMFutures(
            key=os.getenv("BINANCE_API_KEY"),
            secret=os.getenv("BINANCE_API_SECRET"),
            base_url=base_url
        )
        
        # Test public endpoint
        try:
            server_time = client.time()
            check_status(True, f"Connected to {'TESTNET' if use_testnet else 'LIVE'} server")
            
            # Show server time
            server_dt = datetime.fromtimestamp(server_time['serverTime'] / 1000)
            print(f"   Server time: {server_dt}")
            
        except Exception as e:
            check_status(False, f"Failed to connect to Binance: {e}")
            return False
        
        # Test authenticated endpoint
        try:
            account = client.account()
            balance = float(account.get('totalWalletBalance', 0))
            check_status(True, f"Account authenticated - Balance: ${balance:.2f}")
            
            # Check if futures trading is enabled
            if not account.get('canTrade'):
                check_status(False, "Futures trading is not enabled on this account")
                return False
                
        except Exception as e:
            check_status(False, f"Authentication failed: {e}")
            print("   Make sure your API key has Futures permissions")
            return False
            
        return True
        
    except ImportError:
        check_status(False, "binance-connector not installed")
        return False

def test_discord_token():
    """Basic Discord token validation"""
    print_header("Testing Discord Configuration")
    
    token = os.getenv("DISCORD_BOT_TOKEN", "")
    channel_id = os.getenv("DISCORD_CHANNEL_ID", "0")
    
    # Basic token format check
    if len(token) > 50 and "." in token:
        check_status(True, "Discord token format looks valid")
    else:
        check_status(False, "Discord token format appears invalid")
        return False
    
    # Channel ID check
    try:
        channel_id_int = int(channel_id)
        if channel_id_int > 0:
            check_status(True, f"Discord channel ID set: {channel_id_int}")
        else:
            check_status(False, "Discord channel ID not configured")
            return False
    except ValueError:
        check_status(False, "Discord channel ID must be a number")
        return False
    
    print("   Note: Full Discord validation requires running the bot")
    return True

def test_persistence():
    """Test database persistence"""
    print_header("Testing Persistence Layer")
    
    try:
        from main import PersistenceManager
        
        db_file = os.getenv("PERSISTENCE_FILE", "bot_state.db")
        pm = PersistenceManager(db_file)
        
        # Test operations
        test_key = "test_key"
        test_value = {"test": "data", "timestamp": datetime.now().isoformat()}
        
        pm.save_state(test_key, test_value)
        retrieved = pm.get_state(test_key)
        
        if retrieved == test_value:
            check_status(True, "Database persistence working")
            
            # Clean up test data
            import sqlite3
            with sqlite3.connect(db_file) as conn:
                conn.execute("DELETE FROM bot_state WHERE key = ?", (test_key,))
                conn.commit()
                
            return True
        else:
            check_status(False, "Database persistence test failed")
            return False
            
    except Exception as e:
        check_status(False, f"Persistence test error: {e}")
        return False

def test_risk_calculations():
    """Test risk management calculations"""
    print_header("Testing Risk Management")
    
    # Test position sizing logic
    risk_mode = os.getenv("RISK_MODE", "FIXED")
    
    if risk_mode == "PERCENT":
        risk_pct = float(os.getenv("RISK_PERCENT_PER_TRADE", "2"))
        max_pct = float(os.getenv("MAX_POSITION_SIZE_PERCENT", "10"))
        check_status(True, f"Using PERCENT mode: {risk_pct}% risk, {max_pct}% max position")
    else:
        risk_usdt = float(os.getenv("RISK_PER_TRADE_USDT", "10"))
        max_usdt = float(os.getenv("MAX_POSITION_SIZE_USDT", "100"))
        check_status(True, f"Using FIXED mode: ${risk_usdt} risk, ${max_usdt} max position")
    
    # Check safety settings
    min_rr = float(os.getenv("MIN_RISK_REWARD", "2.0"))
    max_slippage = float(os.getenv("MAX_SLIPPAGE_PERCENT", "0.5"))
    
    check_status(min_rr >= 2.0, f"Minimum RR ratio: {min_rr} (recommended ≥ 2.0)")
    check_status(max_slippage <= 1.0, f"Max slippage: {max_slippage}% (recommended ≤ 1.0%)")
    
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "🤖 Trading Bot Setup Validator".center(50))
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Environment", test_environment()))
    results.append(("Libraries", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        results.append(("Signal Parsing", test_signal_parsing()))
        results.append(("Discord", test_discord_token()))
        results.append(("Binance API", test_binance_connection()))
        results.append(("Database", test_persistence()))
        results.append(("Risk Settings", test_risk_calculations()))
    
    # Summary
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name:20} {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your bot is ready to run.")
        print("\nNext steps:")
        print("1. Test in SIMULATION mode first: ENABLE_TRADING=false")
        print("2. Run on TESTNET: USE_TESTNET=true")
        print("3. Monitor with dashboard: python dashboard.py")
        print("4. Only go LIVE after thorough testing!")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the bot.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
