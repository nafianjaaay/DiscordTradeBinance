# Discord to Binance Futures Auto-Trading Bot

⚠️ **IMPORTANT WARNING**: This bot executes real trades with real money. Use at your own risk. Always test thoroughly on testnet first.

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Discord Bot** created with proper permissions
3. **Binance Account** with Futures trading enabled
4. **Binance API Keys** with Futures permissions

## 🚀 Quick Start

### 1. Clone/Download the Bot

```bash
mkdir trading-bot
cd trading-bot
# Copy the main.py file here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example config
cp .env.example .env

# Edit .env with your actual values
nano .env  # or use any text editor
```

### 4. Set Up Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application and bot
3. Copy the bot token to `.env`
4. Invite bot to your server with these permissions:
   - Read Messages
   - Send Messages
   - Read Message History
   - Add Reactions

### 5. Set Up Binance API

#### For Testnet (Recommended for Testing):
1. Go to [Binance Testnet](https://testnet.binancefuture.com/)
2. Create account and generate API keys
3. Add keys to `.env` with `USE_TESTNET=true`

#### For Live Trading:
1. Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Create new API key with **Futures** permissions
3. Enable IP whitelist (recommended)
4. Add keys to `.env` with `USE_TESTNET=false`

### 6. Run the Bot

```bash
# Test mode (simulation)
python main.py

# After thorough testing, enable live trading in .env
# ENABLE_TRADING=true
```

## 📊 Signal Format

The bot expects signals in this format:

```
ATLAS-7 SIGNAL: BTCUSDT LONG
Entry 1: 45000
Entry 2: 44800
Stop Loss: 44000
TP 1: 46000 - 25%
TP 2: 47000 - 35%
TP 3: 48000 - 25%
TP 4: 49000 - 15%
```

## ⚙️ Configuration Guide

### Risk Management Settings

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `RISK_PER_TRADE_USDT` | Max loss per trade | 1-2% of account |
| `MAX_POSITION_SIZE_USDT` | Max position value | 10-20% of account |
| `MAX_OPEN_POSITIONS` | Concurrent positions | 3-5 |
| `MIN_RISK_REWARD` | Minimum RR ratio | 2.0 or higher |
| `DEFAULT_LEVERAGE` | Position leverage | 5x or lower |

### Safety Features

1. **Simulation Mode**: Set `ENABLE_TRADING=false` to test without real trades
2. **Testnet Support**: Use Binance testnet for risk-free testing
3. **Signal Validation**: Checks SL/TP validity and risk-reward ratio
4. **Position Limits**: Prevents over-exposure
5. **Duplicate Prevention**: Won't process same signal twice

## 📁 Project Structure

```
trading-bot/
├── main.py              # Main bot application
├── .env                 # Configuration (create from .env.example)
├── .env.example         # Example configuration
├── requirements.txt     # Python dependencies
├── logs/               # Log files (auto-created)
│   └── trading_bot.log
└── README.md           # This file
```

## 🔍 Monitoring & Logs

### Log Files
- Location: `logs/trading_bot.log`
- Includes all trades, errors, and system events
- Rotate logs periodically to manage size

### Discord Feedback
The bot replies to signals with:
- ✅ Successful execution details
- ❌ Rejection reasons
- Current position information

### Position Monitoring
- Automatic position checking every 5 minutes
- Logs current P&L and exposure

## 🛠️ Troubleshooting

### Common Issues

1. **"Configuration error: XXX not set"**
   - Check your `.env` file has all required values

2. **"Signal validation failed"**
   - Ensure SL is below entries for LONG, above for SHORT
   - Check TPs are in profit direction

3. **"Position sizing failed"**
   - Account balance might be too low
   - Check API permissions include Futures

4. **"Failed to place order"**
   - Verify symbol exists (e.g., BTCUSDT not BTCUSD)
   - Check minimum order size requirements

### Debug Mode

Enable debug logging in `.env`:
```
LOG_LEVEL=DEBUG
```

## ⚠️ Risk Warnings

1. **Start Small**: Test with minimum position sizes
2. **Use Testnet**: Always test strategies on testnet first
3. **Monitor Actively**: Don't leave bot unattended initially
4. **Set Limits**: Use `MAX_POSITION_SIZE_USDT` to limit exposure
5. **Regular Reviews**: Check logs daily for issues
6. **API Security**: 
   - Never share API keys
   - Use IP whitelist
   - Enable 2FA on Binance
   - Restrict API permissions to only what's needed

## 📈 Best Practices

1. **Testing Protocol**:
   - Week 1: Testnet only
   - Week 2: Live with minimum sizes
   - Week 3+: Gradually increase if profitable

2. **Risk Management**:
   - Never risk more than 1-2% per trade
   - Keep total exposure under 10% of account
   - Use stop losses on every trade

3. **Maintenance**:
   - Review logs daily
   - Update dependencies monthly
   - Backup configuration regularly

## 🔄 Updates & Maintenance

### Updating Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Backup Configuration
```bash
cp .env .env.backup
```

### Clear Old Logs
```bash
# Archive old logs
mv logs/trading_bot.log logs/trading_bot_$(date +%Y%m%d).log
```

## 📞 Support & Contributions

- Report issues in the project repository
- Always include logs when reporting problems
- Test modifications on testnet before live deployment

## 📄 License & Disclaimer

**DISCLAIMER**: This software is provided "as is" without warranty of any kind. Trading cryptocurrencies carries substantial risk of loss. The developers assume no responsibility for any losses incurred while using this bot.

**Never invest more than you can afford to lose.**

---

## Quick Command Reference

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env file

# Test (simulation)
ENABLE_TRADING=false python main.py

# Run (after testing)
python main.py

# Run in background (Linux/Mac)
nohup python main.py > output.log 2>&1 &

# Check logs
tail -f logs/trading_bot.log
```

Stay safe and trade responsibly! 🚀
