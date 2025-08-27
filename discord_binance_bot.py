#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord -> Binance Futures Auto-Trading Bot
Monitors Discord for trading signals and executes on Binance Futures
Version: 2.0 - Production Ready with Enhanced Safety Features
"""

import os
import re
import asyncio
import logging
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import time
from dataclasses import dataclass
from enum import Enum

import discord
from discord import Intents
from binance.um_futures import UMFutures
from binance.error import ClientError
from binance.lib.utils import config_logging

# =================== CONFIGURATION ===================

# Load environment variables
from dotenv import load_dotenv
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
RISK_PER_TRADE_USDT = Decimal(os.getenv("RISK_PER_TRADE_USDT", "10"))
MAX_POSITION_SIZE_USDT = Decimal(os.getenv("MAX_POSITION_SIZE_USDT", "100"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

# Safety Features
ENABLE_TRADING = os.getenv("ENABLE_TRADING", "false").lower() == "true"
REQUIRED_SIGNAL_KEYWORD = os.getenv("REQUIRED_SIGNAL_KEYWORD", "ATLAS-7 SIGNAL")
MIN_RISK_REWARD = Decimal(os.getenv("MIN_RISK_REWARD", "2.0"))
MAX_SLIPPAGE_PERCENT = Decimal(os.getenv("MAX_SLIPPAGE_PERCENT", "0.5"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "trading_bot.log")

# =================== LOGGING SETUP ===================

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
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
    LONG = "BUY"
    SHORT = "SELL"

@dataclass
class TradingSignal:
    """Structured representation of a trading signal"""
    symbol: str
    side: OrderSide
    entries: List[Decimal]
    stop_loss: Decimal
    take_profits: List[Tuple[Decimal, int]]  # (price, percentage)
    timestamp: datetime
    raw_message: str
    
    def validate(self) -> bool:
        """Validate signal parameters"""
        # Check if we have valid entries
        if not self.entries or len(self.entries) == 0:
            return False
        
        # Check stop loss validity
        if self.side == OrderSide.LONG:
            if self.stop_loss >= min(self.entries):
                logger.warning(f"Invalid LONG signal: SL {self.stop_loss} >= Entry {min(self.entries)}")
                return False
        else:
            if self.stop_loss <= max(self.entries):
                logger.warning(f"Invalid SHORT signal: SL {self.stop_loss} <= Entry {max(self.entries)}")
                return False
        
        # Check take profits validity
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

# =================== BINANCE CLIENT ===================

class BinanceTrader:
    """Enhanced Binance Futures trading client with safety features"""
    
    def __init__(self):
        base_url = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"
        self.client = UMFutures(
            key=BINANCE_API_KEY,
            secret=BINANCE_API_SECRET,
            base_url=base_url
        )
        self.symbol_info = {}
        self.open_positions = {}
        self.pending_orders = {}
        self._load_exchange_info()
    
    def _load_exchange_info(self):
        """Load and cache exchange trading rules"""
        try:
            info = self.client.exchange_info()
            for symbol in info['symbols']:
                self.symbol_info[symbol['symbol']] = {
                    'pricePrecision': symbol['pricePrecision'],
                    'quantityPrecision': symbol['quantityPrecision'],
                    'minQty': Decimal(next(f['minQty'] for f in symbol['filters'] if f['filterType'] == 'LOT_SIZE')),
                    'minNotional': Decimal(next(f['notional'] for f in symbol['filters'] if f['filterType'] == 'MIN_NOTIONAL')),
                    'tickSize': Decimal(next(f['tickSize'] for f in symbol['filters'] if f['filterType'] == 'PRICE_FILTER'))
                }
            logger.info(f"Loaded exchange info for {len(self.symbol_info)} symbols")
        except Exception as e:
            logger.error(f"Failed to load exchange info: {e}")
    
    def round_price(self, symbol: str, price: Decimal) -> Decimal:
        """Round price to valid tick size"""
        if symbol not in self.symbol_info:
            return price.quantize(Decimal("0.001"))
        
        tick_size = self.symbol_info[symbol]['tickSize']
        return (price / tick_size).quantize(Decimal("1")) * tick_size
    
    def round_quantity(self, symbol: str, qty: Decimal) -> Decimal:
        """Round quantity to valid step size"""
        if symbol not in self.symbol_info:
            return qty.quantize(Decimal("0.001"))
        
        precision = self.symbol_info[symbol]['quantityPrecision']
        return qty.quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN)
    
    def calculate_position_size(self, signal: TradingSignal) -> Optional[Decimal]:
        """Calculate position size based on risk management rules"""
        try:
            # Get current account balance
            account = self.client.account()
            balance = Decimal(account['totalWalletBalance'])
            
            # Check maximum open positions
            positions = [p for p in account['positions'] if Decimal(p['positionAmt']) != 0]
            if len(positions) >= MAX_OPEN_POSITIONS:
                logger.warning(f"Maximum open positions reached: {len(positions)}/{MAX_OPEN_POSITIONS}")
                return None
            
            # Calculate position size based on risk
            avg_entry = sum(signal.entries) / len(signal.entries)
            risk_distance = abs(avg_entry - signal.stop_loss) / avg_entry
            
            # Position size = Risk Amount / Risk Distance
            position_value = RISK_PER_TRADE_USDT / risk_distance
            
            # Apply maximum position size limit
            position_value = min(position_value, MAX_POSITION_SIZE_USDT)
            
            # Calculate quantity
            current_price = self.get_current_price(signal.symbol)
            if not current_price:
                return None
            
            quantity = position_value / current_price
            quantity = self.round_quantity(signal.symbol, quantity)
            
            # Validate minimum notional
            if signal.symbol in self.symbol_info:
                min_notional = self.symbol_info[signal.symbol]['minNotional']
                if quantity * current_price < min_notional:
                    logger.warning(f"Position size below minimum notional: {quantity * current_price} < {min_notional}")
                    return None
            
            return quantity
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price"""
        try:
            ticker = self.client.ticker_price(symbol=symbol)
            return Decimal(ticker['price'])
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol"""
        try:
            self.client.change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False
    
    def place_order_set(self, signal: TradingSignal, quantity: Decimal) -> bool:
        """Place a complete order set (entries, SL, TPs)"""
        if not ENABLE_TRADING:
            logger.info("SIMULATION MODE: Would place orders:")
            logger.info(f"  Symbol: {signal.symbol}")
            logger.info(f"  Side: {signal.side.value}")
            logger.info(f"  Quantity: {quantity}")
            logger.info(f"  Entries: {signal.entries}")
            logger.info(f"  Stop Loss: {signal.stop_loss}")
            logger.info(f"  Take Profits: {signal.take_profits}")
            return True
        
        try:
            # Set leverage
            if not self.set_leverage(signal.symbol, DEFAULT_LEVERAGE):
                return False
            
            # Place entry orders
            entry_orders = []
            qty_per_entry = quantity / len(signal.entries)
            
            for i, entry_price in enumerate(signal.entries):
                try:
                    order = self.client.new_order(
                        symbol=signal.symbol,
                        side=signal.side.value,
                        type="LIMIT",
                        quantity=float(self.round_quantity(signal.symbol, qty_per_entry)),
                        price=float(self.round_price(signal.symbol, entry_price)),
                        timeInForce="GTC",
                        workingType="CONTRACT_PRICE"
                    )
                    entry_orders.append(order)
                    logger.info(f"Placed entry order {i+1}: {order['orderId']}")
                except Exception as e:
                    logger.error(f"Failed to place entry order {i+1}: {e}")
            
            if not entry_orders:
                logger.error("Failed to place any entry orders")
                return False
            
            # Place stop loss order
            sl_side = "SELL" if signal.side == OrderSide.LONG else "BUY"
            try:
                sl_order = self.client.new_order(
                    symbol=signal.symbol,
                    side=sl_side,
                    type="STOP_MARKET",
                    quantity=float(quantity),
                    stopPrice=float(self.round_price(signal.symbol, signal.stop_loss)),
                    workingType="CONTRACT_PRICE",
                    priceProtect=True
                )
                logger.info(f"Placed stop loss order: {sl_order['orderId']}")
            except Exception as e:
                logger.error(f"Failed to place stop loss: {e}")
                # Cancel entry orders if SL fails
                for order in entry_orders:
                    try:
                        self.client.cancel_order(symbol=signal.symbol, orderId=order['orderId'])
                    except:
                        pass
                return False
            
            # Place take profit orders
            tp_side = "SELL" if signal.side == OrderSide.LONG else "BUY"
            remaining_qty = quantity
            
            for i, (tp_price, tp_percent) in enumerate(signal.take_profits):
                tp_qty = quantity * Decimal(tp_percent) / 100
                tp_qty = min(tp_qty, remaining_qty)
                remaining_qty -= tp_qty
                
                if tp_qty <= 0:
                    break
                
                try:
                    tp_order = self.client.new_order(
                        symbol=signal.symbol,
                        side=tp_side,
                        type="LIMIT",
                        quantity=float(self.round_quantity(signal.symbol, tp_qty)),
                        price=float(self.round_price(signal.symbol, tp_price)),
                        timeInForce="GTC",
                        reduceOnly=True,
                        workingType="CONTRACT_PRICE"
                    )
                    logger.info(f"Placed TP order {i+1}: {tp_order['orderId']}")
                except Exception as e:
                    logger.error(f"Failed to place TP {i+1}: {e}")
            
            # Store order info
            self.pending_orders[signal.symbol] = {
                'signal': signal,
                'entry_orders': entry_orders,
                'sl_order': sl_order,
                'timestamp': datetime.now()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to place order set: {e}")
            return False
    
    def check_positions(self):
        """Monitor and log current positions"""
        try:
            account = self.client.account()
            positions = [p for p in account['positions'] if Decimal(p['positionAmt']) != 0]
            
            if positions:
                logger.info(f"Current open positions: {len(positions)}")
                for pos in positions:
                    logger.info(f"  {pos['symbol']}: {pos['positionAmt']} @ {pos['entryPrice']} | PNL: {pos['unrealizedProfit']}")
            
            return positions
        except Exception as e:
            logger.error(f"Failed to check positions: {e}")
            return []

# =================== SIGNAL PARSER ===================

class SignalParser:
    """Enhanced signal parser with validation"""
    
    # Updated regex to handle various signal formats
    SIGNAL_PATTERNS = [
        # Pattern 1: Original format
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
        ),
        # Pattern 2: Alternative format
        re.compile(
            r"(?P<keyword>ATLAS-7\s+SIGNAL)[:\s\-]*"
            r"(?P<pair>[A-Z]{2,10}USDT(?:\.P)?)\s+"
            r"(?P<side>LONG|SHORT)"
            r"[\s\S]*?Entries?[:\s]*\$?(?P<entry1>\d+(?:\.\d+)?)"
            r"(?:[,\s]+\$?(?P<entry2>\d+(?:\.\d+)?))?"
            r"[\s\S]*?SL[:\s]*\$?(?P<sl>\d+(?:\.\d+)?)"
            r"[\s\S]*?Targets?[:\s]*"
            r"\$?(?P<tp1>\d+(?:\.\d+)?)"
            r"(?:[,\s]+\$?(?P<tp2>\d+(?:\.\d+)?))?"
            r"(?:[,\s]+\$?(?P<tp3>\d+(?:\.\d+)?))?"
            r"(?:[,\s]+\$?(?P<tp4>\d+(?:\.\d+)?))?",
            re.IGNORECASE | re.MULTILINE
        )
    ]
    
    @classmethod
    def parse(cls, message: str) -> Optional[TradingSignal]:
        """Parse a message into a TradingSignal"""
        # Check for required keyword
        if REQUIRED_SIGNAL_KEYWORD not in message.upper():
            return None
        
        # Try each pattern
        for pattern in cls.SIGNAL_PATTERNS:
            match = pattern.search(message)
            if match:
                return cls._process_match(match, message)
        
        logger.debug(f"No pattern matched for message: {message[:100]}...")
        return None
    
    @classmethod
    def _process_match(cls, match, raw_message: str) -> Optional[TradingSignal]:
        """Process regex match into TradingSignal"""
        try:
            groups = match.groupdict()
            
            # Extract symbol and side
            symbol = groups['pair'].replace('.P', '')  # Remove .P suffix if present
            side = OrderSide.LONG if groups['side'].upper() == 'LONG' else OrderSide.SHORT
            
            # Extract entries
            entries = []
            for i in range(1, 3):
                entry_key = f'entry{i}'
                if entry_key in groups and groups[entry_key]:
                    entries.append(Decimal(groups[entry_key]))
            
            if not entries:
                logger.warning("No valid entries found")
                return None
            
            # Extract stop loss
            stop_loss = Decimal(groups['sl'])
            
            # Extract take profits with percentages
            take_profits = []
            default_percentages = [25, 35, 25, 15]  # Default TP percentages
            
            for i in range(1, 5):
                tp_key = f'tp{i}'
                tp_pct_key = f'tp{i}_pct'
                
                if tp_key in groups and groups[tp_key]:
                    tp_price = Decimal(groups[tp_key])
                    
                    # Get percentage or use default
                    if tp_pct_key in groups and groups[tp_pct_key]:
                        tp_pct = int(groups[tp_pct_key])
                    else:
                        tp_pct = default_percentages[i-1] if i <= len(default_percentages) else 25
                    
                    take_profits.append((tp_price, tp_pct))
            
            # Create signal object
            signal = TradingSignal(
                symbol=symbol,
                side=side,
                entries=entries,
                stop_loss=stop_loss,
                take_profits=take_profits,
                timestamp=datetime.now(),
                raw_message=raw_message[:500]  # Store first 500 chars
            )
            
            # Validate signal
            if not signal.validate():
                logger.warning("Signal validation failed")
                return None
            
            # Check risk-reward ratio
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
    """Discord bot that monitors signals and executes trades"""
    
    def __init__(self):
        intents = Intents.default()
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)
        
        self.trader = BinanceTrader()
        self.parser = SignalParser()
        self.processed_signals = set()  # Track processed messages to avoid duplicates
        self.last_position_check = datetime.now()
    
    async def on_ready(self):
        """Bot startup"""
        logger.info(f"Bot logged in as {self.user}")
        logger.info(f"Monitoring channel: {DISCORD_CHANNEL_ID}")
        logger.info(f"Trading enabled: {ENABLE_TRADING}")
        logger.info(f"Testnet mode: {USE_TESTNET}")
        
        # Start position monitor task
        asyncio.create_task(self.monitor_positions())
    
    async def on_message(self, message: discord.Message):
        """Process incoming messages"""
        # Filter: correct channel
        if message.channel.id != DISCORD_CHANNEL_ID:
            return
        
        # Filter: not self
        if message.author == self.user:
            return
        
        # Filter: not already processed
        if message.id in self.processed_signals:
            return
        
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
        
        # Skip if no content
        if not content.strip():
            return
        
        # Parse signal
        signal = self.parser.parse(content)
        if not signal:
            return
        
        # Mark as processed
        self.processed_signals.add(message.id)
        
        # Process signal
        await self.process_signal(signal, message)
    
    async def process_signal(self, signal: TradingSignal, message: discord.Message):
        """Process and execute a trading signal"""
        logger.info(f"Processing signal from message {message.id}")
        
        try:
            # Calculate position size
            quantity = self.trader.calculate_position_size(signal)
            if not quantity:
                await message.reply("❌ Signal rejected: Position sizing failed")
                return
            
            # Execute trades
            success = self.trader.place_order_set(signal, quantity)
            
            if success:
                rr = signal.calculate_risk_reward()
                response = (
                    f"✅ **Signal Executed**\n"
                    f"Symbol: {signal.symbol}\n"
                    f"Side: {signal.side.name}\n"
                    f"Quantity: {quantity}\n"
                    f"Risk/Reward: {rr:.2f}\n"
                    f"Mode: {'LIVE' if ENABLE_TRADING else 'SIMULATION'}"
                )
            else:
                response = "❌ **Signal Failed**\nCheck logs for details"
            
            await message.reply(response)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            await message.reply(f"❌ Error: {str(e)[:200]}")
    
    async def monitor_positions(self):
        """Periodic position monitoring task"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check positions every 5 minutes
                if datetime.now() - self.last_position_check > timedelta(minutes=5):
                    positions = self.trader.check_positions()
                    self.last_position_check = datetime.now()
                    
                    # Clean up old processed signals (keep last 100)
                    if len(self.processed_signals) > 100:
                        self.processed_signals = set(list(self.processed_signals)[-100:])
                
            except Exception as e:
                logger.error(f"Position monitor error: {e}")

# =================== MAIN EXECUTION ===================

def validate_config():
    """Validate configuration before starting"""
    errors = []
    
    if not DISCORD_BOT_TOKEN:
        errors.append("DISCORD_BOT_TOKEN not set")
    
    if DISCORD_CHANNEL_ID == 0:
        errors.append("DISCORD_CHANNEL_ID not set")
    
    if not BINANCE_API_KEY:
        errors.append("BINANCE_API_KEY not set")
    
    if not BINANCE_API_SECRET:
        errors.append("BINANCE_API_SECRET not set")
    
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
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
    logger.info("Discord -> Binance Auto-Trading Bot Starting")
    logger.info("=" * 50)
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed. Exiting.")
        return
    
    # Create and run bot
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
