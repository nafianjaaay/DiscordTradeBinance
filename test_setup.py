#!/usr/bin/env python3
"""
Test script to validate CCXT bot setup and configuration
"""

import os
import sys
import json
import ccxt
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
        ("ccxt", "ccxt"),
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
    
    # Check CCXT version
    try:
        import ccxt
        check_status(True, f"CCXT version: {ccxt.__version__}")
    except:
        pass
    
    return all_good

def test_ccxt_connection():
    """Test Binance connection via CCXT"""
    print_header("Testing CCXT Binance Connection")
    
    try:
        use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
        
        # Configure exchange
        exchange_config = {
            'apiKey': os.getenv("BINANCE_API_KEY"),
            'secret': os.getenv("BINANCE_API_SECRET"),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            }
        }
        
        if use_testnet:
            exchange_config['options']['testnet'] = True
            exchange_config['urls'] = {
                'api': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                    'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                    'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                }
            }
        
        exchange = ccxt.binance(exchange_config)
        check_status(True, f"CCXT exchange created for {'TESTNET' if use_testnet else 'LIVE'}")
        
        # Test public endpoint
        try:
            server_time = exchange.fetch_time()
            check_status(True, f"Connected to server - Time: {datetime.fromtimestamp(server_time/1000)}")
        except Exception as e:
            check_status(False, f"Failed to connect: {e}")
            return False
        
        # Load markets
        try:
            markets = exchange.load_markets()
            check_status(True, f"Loaded {len(markets)} markets")
            
            # Show sample market info
            if 'BTC/USDT:USDT' in markets:
                btc_market = markets['BTC/USDT:USDT']
                print(f"   Sample market (BTC/USDT:USDT):")
                print(f"   - Min amount: {btc_market['limits']['amount']['min']}")
                print(f"   - Price precision: {btc_market['precision']['price']}")
                print(f"   - Amount precision: {btc_market['precision']['amount']}")
        except Exception as e:
            check_status(False, f"Failed to load markets: {e}")
            return False
        
        # Test authenticated endpoints
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['USDT']['total'] if 'USDT' in balance else 0
            check_status(True, f"Account authenticated - Balance: ${usdt_balance:.2f} USDT")
        except Exception as e:
            check_status(False, f"Authentication failed: {e}")
            print("   Make sure your API key has Futures permissions")
            return False
        
        # Test futures-specific functions
        try:
            positions = exchange.fetch_positions()
            open_positions = [p for p in positions if p['contracts'] != 0]
            check_status(True, f"Fetched positions - {len(open_positions)} open")
        except Exception as e:
            check_status(False, f"Failed to fetch positions: {e}")
            return False
        
        return True
        
    except ImportError:
        check_status(False, "CCXT not installed")
        return False
    except Exception as e:
        check_status(False, f"Unexpected error: {e}")
        return False

def test_signal_parsing():
    """Test signal parser with CCXT format"""
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
        # Check CCXT symbol format conversion
        check_status(True, f"Signal parsed: {signal.symbol}")
        check_status('/USDT:USDT' in signal.symbol, f"Symbol converted to CCXT format: {signal.symbol}")
        check_status(signal.validate(), "Signal validation passed")
        rr = signal.calculate_risk_reward()
        check_status(rr > 0, f"Risk-Reward calculated: {rr:.2f}")
        return True
    else:
        check_status(False, "Failed to parse test signal")
        return False

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

def test_ccxt_order_placement():
    """Test order placement functionality (dry run)"""
    print_header("Testing Order Placement (Dry Run)")
    
    try:
        use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
        
        if not use_testnet:
            print("⚠️  Skipping order placement test on LIVE account")
            return True
        
        exchange_config = {
            'apiKey': os.getenv("BINANCE_API_KEY"),
            'secret': os.getenv("BINANCE_API_SECRET"),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'testnet': True
            },
            'urls': {
                'api': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                    'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                    'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                }
            }
        }
        
        exchange = ccxt.binance(exchange_config)
        exchange.load_markets()
        
        # Test symbol
        symbol = 'BTC/USDT:USDT'
        
        if symbol not in exchange.markets:
            print(f"   Test symbol {symbol} not found in markets")
            return True
        
        # Get current price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        check_status(True, f"Current {symbol} price: ${current_price:.2f}")
        
        # Test order parameters
        market = exchange.markets[symbol]
        min_amount = market['limits']['amount']['min']
        
        print(f"   Order parameters for {symbol}:")
        print(f"   - Min order size: {min_amount}")
        print(f"   - Price precision: {market['precision']['price']}")
        print(f"   - Amount precision: {market['precision']['amount']}")
        
        check_status(True, "Order placement parameters validated")
        
        print("\n   Note: Actual order placement skipped (dry run only)")
        print("   To test real orders, use the bot in SIMULATION mode first")
        
        return True
        
    except Exception as e:
        check_status(False, f"Order test error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "🤖 Trading Bot CCXT Setup Validator".center(50))
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Environment", test_environment()))
    results.append(("Libraries", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        results.append(("CCXT Connection", test_ccxt_connection()))
        results.append(("Signal Parsing", test_signal_parsing()))
        results.append(("Database", test_persistence()))
        results.append(("Risk Settings", test_risk_calculations()))
        results.append(("Order Placement", test_ccxt_order_placement()))
    
    # Summary
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name:20} {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your CCXT bot is ready to run.")
        print("\nNext steps:")
        print("1. Test in SIMULATION mode first: ENABLE_TRADING=false")
        print("2. Run on TESTNET: USE_TESTNET=true")
        print("3. Monitor with dashboard: python dashboard_ccxt.py")
        print("4. Only go LIVE after thorough testing!")
        print("\n💡 CCXT advantages:")
        print("   - Better maintained and documented")
        print("   - Supports 100+ exchanges")
        print("   - Unified API across all exchanges")
        print("   - Active community support")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the bot.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
