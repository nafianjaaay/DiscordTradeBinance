#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord -> Binance Futures Auto-Trading Bot
Version: 3.0 - Using CCXT library for better reliability
"""

import os
import re
import asyncio
import logging
import json
import sqlite3
import ccxt
import ccxt.async_support as ccxt_async
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import discord
from discord import Intents
from dotenv import load_dotenv

# =================== CONFIGURATION ===================
load_dotenv()

# Discord Configuration
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))

# Binance Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"

# Trading Parameters
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "5"))

# Risk Management
RISK_MODE = os.getenv("RISK_MODE", "FIXED")  # FIXED or PERCENT
RISK_PER_TRADE_USDT = Decimal(os.getenv("RISK_PER_TRADE_USDT", "10"))
RISK_PERCENT_PER_TRADE = Decimal(os.getenv("RISK_PERCENT_PER_TRADE", "2"))
MAX_POSITION_SIZE_USDT = Decimal(os.getenv("MAX_POSITION_SIZE_USDT", "100"))
MAX_POSITION_SIZE_PERCENT = Decimal(os.getenv("MAX_POSITION_SIZE_PERCENT", "10"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

# Safety Features
ENABLE_TRADING = os.getenv("ENABLE_TRADING", "false").lower() == "true"
REQUIRED_SIGNAL_KEYWORD = os.getenv("REQUIRED_SIGNAL_KEYWORD", "ATLAS-7 SIGNAL")
MIN_RISK_REWARD = Decimal(os.getenv("MIN_RISK_REWARD", "2.0"))
MAX_SLIPPAGE_PERCENT = Decimal(os.getenv("MAX_SLIPPAGE_PERCENT", "0.5"))

# Persistence
PERSISTENCE_FILE = os.getenv("PERSISTENCE_FILE", "bot_state.db")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "trading_bot.log")

# =================== LOGGING SETUP ===================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{LOG_FILE}"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =================== DATA STRUCTURES ===================

class OrderSide(Enum):
    LONG = "buy"  # CCXT uses lowercase
    SHORT = "sell"

@dataclass
class TradingSignal:
    """Structured representation of a trading signal"""
    symbol: str
    side: OrderSide
    entries: List[Decimal]
    stop_loss: Decimal
    take_profits: List[Tuple[Decimal, int]]
    timestamp: datetime
    raw_message: str
    message_id: int = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for persistence"""
        return {
            'symbol': self.symbol,
            'side': self.side.name,
            'entries': [str(e) for e in self.entries],
            'stop_loss': str(self.stop_loss),
            'take_profits': [(str(tp), pct) for tp, pct in self.take_profits],
            'timestamp': self.timestamp.isoformat(),
            'raw_message': self.raw_message,
            'message_id': self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TradingSignal':
        """Create from dictionary"""
        return cls(
            symbol=data['symbol'],
            side=OrderSide[data['side']],
            entries=[Decimal(e) for e in data['entries']],
            stop_loss=Decimal(data['stop_loss']),
            take_profits=[(Decimal(tp), pct) for tp, pct in data['take_profits']],
            timestamp=datetime.fromisoformat(data['timestamp']),
            raw_message=data['raw_message'],
            message_id=data.get('message_id')
        )
    
    def validate(self) -> bool:
        """Validate signal parameters"""
        if not self.entries or len(self.entries) == 0:
            return False
        
        if self.side == OrderSide.LONG:
            if self.stop_loss >= min(self.entries):
                logger.warning(f"Invalid LONG signal: SL {self.stop_loss} >= Entry {min(self.entries)}")
                return False
        else:
            if self.stop_loss <= max(self.entries):
                logger.warning(f"Invalid SHORT signal: SL {self.stop_loss} <= Entry {max(self.entries)}")
                return False
        
        for tp, _ in self.take_profits:
            if self.side == OrderSide.LONG:
                if tp <= max(self.entries):
                    logger.warning(f"Invalid LONG TP: {tp} <= Entry {max(self.entries)}")
                    return False
            else:
                if tp >= min(self.entries):
                    logger.warning(f"Invalid SHORT TP: {tp} >= Entry {min(self.entries)}")
                    return False
        
        return True
    
    def calculate_risk_reward(self) -> Decimal:
        """Calculate average risk-reward ratio"""
        avg_entry = sum(self.entries) / len(self.entries)
        risk = abs(avg_entry - self.stop_loss)
        
        if not self.take_profits:
            return Decimal("0")
        
        avg_tp = sum(tp for tp, _ in self.take_profits) / len(self.take_profits)
        reward = abs(avg_tp - avg_entry)
        
        return reward / risk if risk > 0 else Decimal("0")

# =================== PERSISTENCE MANAGER ===================

class PersistenceManager:
    """Manages bot state persistence using SQLite"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_signals (
                    message_id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    timestamp TEXT,
                    processed_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    type TEXT,
                    quantity TEXT,
                    price TEXT,
                    status TEXT,
                    created_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()
    
    def is_signal_processed(self, message_id: int) -> bool:
        """Check if a signal has been processed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM processed_signals WHERE message_id = ?",
                (message_id,)
            )
            return cursor.fetchone() is not None
    
    def mark_signal_processed(self, signal: TradingSignal):
        """Mark a signal as processed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processed_signals 
                (message_id, symbol, side, timestamp, processed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                signal.message_id,
                signal.symbol,
                signal.side.name,
                signal.timestamp.isoformat(),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def clean_old_signals(self, days: int = 7):
        """Remove signals older than specified days"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM processed_signals WHERE processed_at < ?",
                (cutoff,)
            )
            conn.commit()
    
    def save_state(self, key: str, value: Any):
        """Save arbitrary state data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO bot_state (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), datetime.now().isoformat()))
            conn.commit()
    
    def get_state(self, key: str, default=None) -> Any:
        """Get state data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM bot_state WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return default

# =================== CCXT BINANCE CLIENT ===================

class BinanceTrader:
    """Binance Futures trading client using CCXT"""
    
    def __init__(self):
        # Configure exchange
        exchange_config = {
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use futures market
                'adjustForTimeDifference': True,
            }
        }
        
        if USE_TESTNET:
            exchange_config['options']['defaultType'] = 'future'
            exchange_config['options']['testnet'] = True
            # CCXT testnet URLs for Binance
            exchange_config['urls'] = {
                'api': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                    'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                    'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                }
            }
        
        self.exchange = ccxt.binance(exchange_config)
        self.markets = {}
        self.open_positions = {}
        self.pending_orders = {}
        self._load_markets_with_retry()
    
    def _load_markets_with_retry(self, max_retries: int = 5):
        """Load market info with retry logic"""
        for attempt in range(max_retries):
            try:
                self.markets = self.exchange.load_markets()
                logger.info(f"Loaded {len(self.markets)} markets from exchange")
                return
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed to load markets: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.critical("Failed to load markets after all retries")
        raise RuntimeError("Cannot proceed without market info")
    
    def get_symbol_info(self, symbol: str) -> dict:
        """Get symbol trading rules"""
        if symbol in self.markets:
            market = self.markets[symbol]
            return {
                'min_qty': market['limits']['amount']['min'],
                'max_qty': market['limits']['amount']['max'],
                'qty_precision': market['precision']['amount'],
                'price_precision': market['precision']['price'],
                'min_notional': market['limits']['cost']['min'] if 'cost' in market['limits'] else 10,
                'tick_size': market['precision']['price']
            }
        return None
    
    def round_price(self, symbol: str, price: Decimal) -> float:
        """Round price to valid precision"""
        if symbol in self.markets:
            precision = self.markets[symbol]['precision']['price']
            if isinstance(precision, int):
                return float(Decimal(str(price)).quantize(Decimal(10) ** -precision))
            else:
                # If precision is a tick size
                tick = Decimal(str(precision))
                return float((Decimal(str(price)) / tick).quantize(Decimal('1')) * tick)
        return float(price)
    
    def round_quantity(self, symbol: str, qty: Decimal) -> float:
        """Round quantity to valid precision"""
        if symbol in self.markets:
            precision = self.markets[symbol]['precision']['amount']
            if isinstance(precision, int):
                return float(Decimal(str(qty)).quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN))
            else:
                # If precision is a step size
                step = Decimal(str(precision))
                return float((Decimal(str(qty)) / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step)
        return float(qty)
    
    def get_account_balance(self) -> Optional[Decimal]:
        """Get current USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['total'] if 'USDT' in balance else 0
            return Decimal(str(usdt_balance))
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return None
    
    def get_open_positions(self) -> List[dict]:
        """Get all open positions"""
        try:
            positions = self.exchange.fetch_positions()
            return [p for p in positions if p['contracts'] != 0]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def calculate_position_size(self, signal: TradingSignal) -> Optional[Decimal]:
        """Calculate position size with dynamic risk management"""
        try:
            # Get account balance
            balance = self.get_account_balance()
            if not balance:
                logger.error("Could not get account balance")
                return None
            
            # Check maximum open positions
            open_positions = self.get_open_positions()
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                logger.warning(f"Maximum open positions reached: {len(open_positions)}/{MAX_OPEN_POSITIONS}")
                return None
            
            # Calculate risk amount based on mode
            if RISK_MODE == "PERCENT":
                risk_amount = balance * RISK_PERCENT_PER_TRADE / 100
                max_position_value = balance * MAX_POSITION_SIZE_PERCENT / 100
            else:
                risk_amount = RISK_PER_TRADE_USDT
                max_position_value = MAX_POSITION_SIZE_USDT
            
            # Calculate position size based on risk
            avg_entry = sum(signal.entries) / len(signal.entries)
            risk_distance = abs(avg_entry - signal.stop_loss) / avg_entry
            
            # Position size = Risk Amount / Risk Distance
            position_value = risk_amount / risk_distance
            
            # Apply maximum position size limit
            position_value = min(position_value, max_position_value)
            
            # Get current price and check slippage
            ticker = self.exchange.fetch_ticker(signal.symbol)
            current_price = Decimal(str(ticker['last']))
            
            # Check slippage
            slippage = abs((current_price - avg_entry) / avg_entry * 100)
            if slippage > MAX_SLIPPAGE_PERCENT:
                logger.warning(f"Slippage too high: {slippage:.2f}% > {MAX_SLIPPAGE_PERCENT}%")
                return None
            
            # Calculate quantity
            quantity = position_value / current_price
            
            # Get symbol info and validate
            symbol_info = self.get_symbol_info(signal.symbol)
            if symbol_info:
                min_qty = Decimal(str(symbol_info['min_qty']))
                if quantity < min_qty:
                    logger.warning(f"Position size below minimum: {quantity} < {min_qty}")
                    return None
                
                # Check minimum notional
                min_notional = Decimal(str(symbol_info['min_notional']))
                if quantity * current_price < min_notional:
                    logger.warning(f"Position value below minimum notional: {quantity * current_price} < {min_notional}")
                    return None
            
            logger.info(f"Position sizing: Risk ${risk_amount:.2f} | Size: {quantity:.4f} @ ${current_price:.2f}")
            return quantity
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return None
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol"""
        try:
            # CCXT method for setting leverage
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False
    
    def place_order_set(self, signal: TradingSignal, quantity: Decimal) -> dict:
        """Place complete order set with CCXT"""
        if not ENABLE_TRADING:
            logger.info("SIMULATION MODE: Would place orders:")
            logger.info(f"  Symbol: {signal.symbol}")
            logger.info(f"  Side: {signal.side.value}")
            logger.info(f"  Quantity: {quantity}")
            logger.info(f"  Entries: {signal.entries}")
            logger.info(f"  Stop Loss: {signal.stop_loss}")
            logger.info(f"  Take Profits: {signal.take_profits}")
            return {'success': True, 'simulation': True}
        
        result = {
            'success': False,
            'entry_orders': [],
            'filled_quantity': Decimal("0"),
            'sl_order': None,
            'tp_orders': []
        }
        
        try:
            # Set leverage
            if not self.set_leverage(signal.symbol, DEFAULT_LEVERAGE):
                return result
            
            # Convert quantity to float for CCXT
            total_qty = self.round_quantity(signal.symbol, quantity)
            
            # Place entry orders
            qty_per_entry = total_qty / len(signal.entries)
            
            for i, entry_price in enumerate(signal.entries):
                try:
                    order_side = signal.side.value
                    order = self.exchange.create_order(
                        symbol=signal.symbol,
                        type='limit',
                        side=order_side,
                        amount=qty_per_entry,
                        price=self.round_price(signal.symbol, entry_price),
                        params={'timeInForce': 'GTC'}
                    )
                    result['entry_orders'].append(order)
                    logger.info(f"Placed entry order {i+1}: {order['id']}")
                    
                    # Check if filled
                    time.sleep(0.5)
                    order_status = self.exchange.fetch_order(order['id'], signal.symbol)
                    if order_status['status'] == 'closed':
                        result['filled_quantity'] += Decimal(str(order_status['filled']))
                        
                except Exception as e:
                    logger.error(f"Failed to place entry order {i+1}: {e}")
            
            if not result['entry_orders']:
                logger.error("Failed to place any entry orders")
                return result
            
            # Use actual filled quantity or total for SL/TP
            actual_qty = float(result['filled_quantity']) if result['filled_quantity'] > 0 else total_qty
            
            # Place stop loss
            sl_side = 'sell' if signal.side == OrderSide.LONG else 'buy'
            try:
                sl_order = self.exchange.create_order(
                    symbol=signal.symbol,
                    type='stop_market',
                    side=sl_side,
                    amount=actual_qty,
                    stopPrice=self.round_price(signal.symbol, signal.stop_loss),
                    params={'stopPrice': self.round_price(signal.symbol, signal.stop_loss)}
                )
                result['sl_order'] = sl_order
                logger.info(f"Placed stop loss order: {sl_order['id']}")
            except Exception as e:
                logger.error(f"Failed to place stop loss: {e}")
                # Cancel entry orders if SL fails
                self._cancel_orders(signal.symbol, result['entry_orders'])
                return result
            
            # Place take profit orders
            tp_side = 'sell' if signal.side == OrderSide.LONG else 'buy'
            remaining_qty = actual_qty
            
            for i, (tp_price, tp_percent) in enumerate(signal.take_profits):
                tp_qty = actual_qty * (tp_percent / 100)
                tp_qty = min(tp_qty, remaining_qty)
                remaining_qty -= tp_qty
                
                if tp_qty <= 0:
                    break
                
                try:
                    tp_order = self.exchange.create_order(
                        symbol=signal.symbol,
                        type='limit',
                        side=tp_side,
                        amount=self.round_quantity(signal.symbol, Decimal(str(tp_qty))),
                        price=self.round_price(signal.symbol, tp_price),
                        params={'timeInForce': 'GTC', 'reduceOnly': True}
                    )
                    result['tp_orders'].append(tp_order)
                    logger.info(f"Placed TP order {i+1}: {tp_order['id']}")
                except Exception as e:
                    logger.error(f"Failed to place TP {i+1}: {e}")
            
            result['success'] = True
            return result
            
        except Exception as e:
            logger.error(f"Failed to place order set: {e}")
            return result
    
    def _cancel_orders(self, symbol: str, orders: List[dict]):
        """Cancel multiple orders"""
        for order in orders:
            try:
                self.exchange.cancel_order(order['id'], symbol)
                logger.info(f"Cancelled order {order['id']}")
            except Exception as e:
                logger.error(f"Failed to cancel order {order['id']}: {e}")
    
    def check_positions(self) -> List[dict]:
        """Get and log current positions"""
        try:
            positions = self.get_open_positions()
            
            if positions:
                logger.info(f"Current open positions: {len(positions)}")
                for pos in positions:
                    pnl = pos['unrealizedPnl'] if 'unrealizedPnl' in pos else 0
                    logger.info(
                        f"  {pos['symbol']}: {pos['contracts']} @ {pos['markPrice']} | "
                        f"PNL: ${pnl:.2f}"
                    )
            
            return positions
        except Exception as e:
            logger.error(f"Failed to check positions: {e}")
            return []

# =================== SIGNAL PARSER ===================

class SignalParser:
    """Enhanced signal parser with validation"""
    
    SIGNAL_PATTERNS = [
        re.compile(
            r"(?P<keyword>ATLAS-7\s+SIGNAL)[:\s\-]*"
            r"(?P<pair>[A-Z]{2,10}USDT(?:\.P)?)\s+"
            r"(?P<side>LONG|SHORT)"
            r"[\s\S]*?Entry\s*1[:\s]*\$?(?P<entry1>\d+(?:\.\d+)?)"
            r"[\s\S]*?Entry\s*2[:\s]*\$?(?P<entry2>\d+(?:\.\d+)?)"
            r"[\s\S]*?Stop\s*Loss[:\s]*\$?(?P<sl>\d+(?:\.\d+)?)"
            r"(?:[\s\S]*?TP\s*1[:\s]*\$?(?P<tp1>\d+(?:\.\d+)?)[^\d]*(?P<tp1_pct>\d+)%?)?"
            r"(?:[\s\S]*?TP\s*2[:\s]*\$?(?P<tp2>\d+(?:\.\d+)?)[^\d]*(?P<tp2_pct>\d+)%?)?"
            r"(?:[\s\S]*?TP\s*3[:\s]*\$?(?P<tp3>\d+(?:\.\d+)?)[^\d]*(?P<tp3_pct>\d+)%?)?"
            r"(?:[\s\S]*?TP\s*4[:\s]*\$?(?P<tp4>\d+(?:\.\d+)?)[^\d]*(?P<tp4_pct>\d+)%?)?",
            re.IGNORECASE | re.MULTILINE
        )
    ]
    
    @classmethod
    def parse(cls, message_content: str, message_id: int = None) -> Optional[TradingSignal]:
        """Parse a message into a TradingSignal"""
        if REQUIRED_SIGNAL_KEYWORD not in message_content.upper():
            return None
        
        for pattern in cls.SIGNAL_PATTERNS:
            match = pattern.search(message_content)
            if match:
                signal = cls._process_match(match, message_content)
                if signal:
                    signal.message_id = message_id
                    # Convert symbol format for CCXT (remove .P suffix, add / for spot symbols)
                    signal.symbol = signal.symbol.replace('.P', '')
                    # For futures, CCXT uses format like 'BTC/USDT:USDT'
                    if not '/' in signal.symbol:
                        base = signal.symbol.replace('USDT', '')
                        signal.symbol = f"{base}/USDT:USDT"
                return signal
        
        logger.debug(f"No pattern matched for message: {message_content[:100]}...")
        return None
    
    @classmethod
    def _process_match(cls, match, raw_message: str) -> Optional[TradingSignal]:
        """Process regex match into TradingSignal"""
        try:
            groups = match.groupdict()
            
            symbol = groups['pair'].replace('.P', '')
            side = OrderSide.LONG if groups['side'].upper() == 'LONG' else OrderSide.SHORT
            
            entries = []
            for i in range(1, 3):
                entry_key = f'entry{i}'
                if entry_key in groups and groups[entry_key]:
                    entries.append(Decimal(groups[entry_key]))
            
            if not entries:
                logger.warning("No valid entries found")
                return None
            
            stop_loss = Decimal(groups['sl'])
            
            take_profits = []
            default_percentages = [25, 35, 25, 15]
            
            for i in range(1, 5):
                tp_key = f'tp{i}'
                tp_pct_key = f'tp{i}_pct'
                
                if tp_key in groups and groups[tp_key]:
                    tp_price = Decimal(groups[tp_key])
                    
                    if tp_pct_key in groups and groups[tp_pct_key]:
                        tp_pct = int(groups[tp_pct_key])
                    else:
                        tp_pct = default_percentages[i-1] if i <= len(default_percentages) else 25
                    
                    take_profits.append((tp_price, tp_pct))
            
            signal = TradingSignal(
                symbol=symbol,
                side=side,
                entries=entries,
                stop_loss=stop_loss,
                take_profits=take_profits,
                timestamp=datetime.now(),
                raw_message=raw_message[:500]
            )
            
            if not signal.validate():
                logger.warning("Signal validation failed")
                return None
            
            rr = signal.calculate_risk_reward()
            if rr < MIN_RISK_REWARD:
                logger.warning(f"Risk-reward ratio too low: {rr} < {MIN_RISK_REWARD}")
                return None
            
            logger.info(f"Successfully parsed signal: {symbol} {side.name} | RR: {rr:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error processing match: {e}")
            return None

# =================== DISCORD BOT ===================

class TradingBot(discord.Client):
    """Discord bot with concurrency control and persistence"""
    
    def __init__(self):
        intents = Intents.default()
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)
        
        self.trader = BinanceTrader()
        self.parser = SignalParser()
        self.persistence = PersistenceManager(PERSISTENCE_FILE)
        self.last_position_check = datetime.now()
        
        # Concurrency control
        self.order_lock = asyncio.Lock()
        self.processing_signals = set()
    
    async def on_ready(self):
        """Bot startup"""
        logger.info(f"Bot logged in as {self.user}")
        logger.info(f"Monitoring channel: {DISCORD_CHANNEL_ID}")
        logger.info(f"Trading enabled: {ENABLE_TRADING}")
        logger.info(f"Testnet mode: {USE_TESTNET}")
        logger.info(f"Risk mode: {RISK_MODE}")
        logger.info(f"Exchange: Binance (via CCXT)")
        
        # Show current balance
        balance = self.trader.get_account_balance()
        if balance:
            logger.info(f"Account balance: ${balance:.2f}")
        
        # Start background tasks
        asyncio.create_task(self.monitor_positions())
        asyncio.create_task(self.cleanup_old_data())
    
    async def on_message(self, message: discord.Message):
        """Process incoming messages"""
        # Filters
        if message.channel.id != DISCORD_CHANNEL_ID:
            return
        
        if message.author == self.user:
            return
        
        # Check persistence for duplicate
        if self.persistence.is_signal_processed(message.id):
            logger.debug(f"Message {message.id} already processed")
            return
        
        # Check if currently processing this signal
        if message.id in self.processing_signals:
            return
        
        # Mark as processing
        self.processing_signals.add(message.id)
        
        try:
            # Combine message content and embeds
            content = message.content or ""
            
            if message.embeds:
                for embed in message.embeds:
                    if embed.title:
                        content += f"\n{embed.title}"
                    if embed.description:
                        content += f"\n{embed.description}"
                    for field in embed.fields:
                        content += f"\n{field.name}: {field.value}"
            
            if not content.strip():
                return
            
            # Parse signal
            signal = self.parser.parse(content, message.id)
            if not signal:
                return
            
            # Process signal with lock
            await self.process_signal(signal, message)
            
        finally:
            # Remove from processing set
            self.processing_signals.discard(message.id)
    
    async def process_signal(self, signal: TradingSignal, message: discord.Message):
        """Process signal with concurrency control"""
        logger.info(f"Processing signal from message {message.id}")
        
        # Use lock to prevent concurrent order placement
        async with self.order_lock:
            try:
                # Mark as processed immediately to prevent duplicates
                self.persistence.mark_signal_processed(signal)
                
                # Calculate position size
                quantity = self.trader.calculate_position_size(signal)
                if not quantity:
                    await message.reply("❌ Signal rejected: Position sizing or slippage check failed")
                    return
                
                # Execute trades
                result = self.trader.place_order_set(signal, quantity)
                
                if result['success']:
                    rr = signal.calculate_risk_reward()
                    
                    # Get account balance for context
                    balance = self.trader.get_account_balance()
                    balance_str = f" | Balance: ${balance:.2f}" if balance else ""
                    
                    # Format symbol for display (remove :USDT suffix)
                    display_symbol = signal.symbol.replace(':USDT', '')
                    
                    response = (
                        f"✅ **Signal Executed**\n"
                        f"Symbol: {display_symbol}\n"
                        f"Side: {signal.side.name}\n"
                        f"Quantity: {quantity:.4f}\n"
                        f"Risk/Reward: {rr:.2f}\n"
                        f"Mode: {'LIVE' if ENABLE_TRADING else 'SIMULATION'}"
                        f"{balance_str}"
                    )
                    
                    if 'filled_quantity' in result and result['filled_quantity'] > 0:
                        response += f"\nFilled: {result['filled_quantity']:.4f}/{quantity:.4f}"
                else:
                    response = "❌ **Signal Failed**\nCheck logs for details"
                
                await message.reply(response)
                
            except Exception as e:
                logger.error(f"Error processing signal: {e}")
                await message.reply(f"❌ Error: {str(e)[:200]}")
    
    async def monitor_positions(self):
        """Periodic position monitoring"""
        while True:
            try:
                await asyncio.sleep(60)
                
                if datetime.now() - self.last_position_check > timedelta(minutes=5):
                    positions = self.trader.check_positions()
                    self.last_position_check = datetime.now()
                    
                    # Save current position count
                    self.persistence.save_state('position_count', len(positions))
                
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
    
    async def cleanup_old_data(self):
        """Periodic cleanup of old data"""
        while True:
            try:
                await asyncio.sleep(86400)  # Daily
                self.persistence.clean_old_signals(days=7)
                logger.info("Cleaned up old signal data")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

# =================== MAIN EXECUTION ===================

def validate_config():
    """Validate configuration"""
    errors = []
    
    if not DISCORD_BOT_TOKEN:
        errors.append("DISCORD_BOT_TOKEN not set")
    
    if DISCORD_CHANNEL_ID == 0:
        errors.append("DISCORD_CHANNEL_ID not set")
    
    if not BINANCE_API_KEY:
        errors.append("BINANCE_API_KEY not set")
    
    if not BINANCE_API_SECRET:
        errors.append("BINANCE_API_SECRET not set")
    
    if RISK_MODE not in ["FIXED", "PERCENT"]:
        errors.append("RISK_MODE must be FIXED or PERCENT")
    
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return False
    
    # Test CCXT installation
    try:
        import ccxt
        logger.info(f"CCXT version: {ccxt.__version__}")
    except ImportError:
        logger.error("CCXT not installed. Run: pip install ccxt")
        return False
    
    # Warnings
    if not ENABLE_TRADING:
        logger.warning("TRADING IS DISABLED - Running in simulation mode")
    
    if not USE_TESTNET and ENABLE_TRADING:
        logger.warning("USING LIVE BINANCE - Real money at risk!")
    
    return True

def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("Discord -> Binance Auto-Trading Bot v3.0 (CCXT)")
    logger.info("=" * 50)
    
    if not validate_config():
        logger.error("Configuration validation failed. Exiting.")
        return
    
    try:
        bot = TradingBot()
        bot.run(DISCORD_BOT_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
