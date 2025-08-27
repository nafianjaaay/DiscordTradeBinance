#!/usr/bin/env python3
"""
Web Dashboard for Discord-Binance Trading Bot
Provides real-time monitoring via a simple web interface
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from decimal import Decimal
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import threading
import time

from binance.um_futures import UMFutures
from dotenv import load_dotenv

# Load configuration
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
PERSISTENCE_FILE = os.getenv("PERSISTENCE_FILE", "bot_state.db")
LOG_FILE = f"logs/{os.getenv('LOG_FILE', 'trading_bot.log')}"
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Initialize Binance client
base_url = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"
binance_client = UMFutures(
    key=BINANCE_API_KEY,
    secret=BINANCE_API_SECRET,
    base_url=base_url
) if BINANCE_API_KEY else None

# HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        .card h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .stat {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .stat:last-child { border-bottom: none; }
        .label { color: #666; }
        .value { 
            font-weight: 600;
            color: #333;
        }
        .value.green { color: #10b981; }
        .value.red { color: #ef4444; }
        .value.yellow { color: #f59e0b; }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .status-online { 
            background: #10b98120;
            color: #10b981;
        }
        .status-offline { 
            background: #ef444420;
            color: #ef4444;
        }
        .position-row {
            background: #f9fafb;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .position-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .position-details {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            font-size: 0.9em;
        }
        .log-viewer {
            background: #1a1a1a;
            color: #0f0;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            max-height: 400px;
            overflow-y: auto;
            line-height: 1.5;
        }
        .log-line {
            margin: 2px 0;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .log-line.error { color: #ff6b6b; }
        .log-line.warning { color: #ffd93d; }
        .log-line.info { color: #6bcf7f; }
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            margin: 0 auto;
            display: block;
            transition: background 0.2s;
        }
        .refresh-btn:hover {
            background: #5a67d8;
        }
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .chart {
            height: 200px;
            margin-top: 15px;
            position: relative;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Trading Bot Dashboard</h1>
        
        <div class="grid">
            <!-- Status Card -->
            <div class="card">
                <h2>📊 System Status</h2>
                <div id="status-content">
                    <div class="stat">
                        <span class="label">Bot Status</span>
                        <span class="value"><span class="spinner"></span></span>
                    </div>
                </div>
            </div>
            
            <!-- Account Card -->
            <div class="card">
                <h2>💰 Account Info</h2>
                <div id="account-content">
                    <div class="stat">
                        <span class="label">Loading...</span>
                        <span class="value"><span class="spinner"></span></span>
                    </div>
                </div>
            </div>
            
            <!-- Statistics Card -->
            <div class="card">
                <h2>📈 Statistics</h2>
                <div id="stats-content">
                    <div class="stat">
                        <span class="label">Loading...</span>
                        <span class="value"><span class="spinner"></span></span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Positions Card -->
        <div class="card">
            <h2>🎯 Open Positions</h2>
            <div id="positions-content">
                <p style="color: #999; text-align: center;">Loading...</p>
            </div>
        </div>
        
        <!-- Recent Signals Card -->
        <div class="card">
            <h2>📡 Recent Signals</h2>
            <div id="signals-content">
                <p style="color: #999; text-align: center;">Loading...</p>
            </div>
        </div>
        
        <!-- Logs Card -->
        <div class="card">
            <h2>📝 Recent Logs</h2>
            <div class="log-viewer" id="logs-content">
                Loading logs...
            </div>
        </div>
        
        <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
    </div>
    
    <script>
        let autoRefresh = true;
        
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        function updateDashboard(data) {
            // Update Status
            const statusHtml = `
                <div class="stat">
                    <span class="label">Bot Status</span>
                    <span class="value">
                        <span class="status-badge ${data.bot_running ? 'status-online' : 'status-offline'}">
                            ${data.bot_running ? '● ONLINE' : '○ OFFLINE'}
                        </span>
                    </span>
                </div>
                <div class="stat">
                    <span class="label">Trading Mode</span>
                    <span class="value">${data.trading_mode}</span>
                </div>
                <div class="stat">
                    <span class="label">Environment</span>
                    <span class="value">${data.testnet ? 'TESTNET' : 'LIVE'}</span>
                </div>
                <div class="stat">
                    <span class="label">Last Update</span>
                    <span class="value">${new Date().toLocaleTimeString()}</span>
                </div>
            `;
            document.getElementById('status-content').innerHTML = statusHtml;
            
            // Update Account
            const accountHtml = `
                <div class="stat">
                    <span class="label">Balance</span>
                    <span class="value">$${data.balance.toFixed(2)}</span>
                </div>
                <div class="stat">
                    <span class="label">Unrealized PNL</span>
                    <span class="value ${data.unrealized_pnl >= 0 ? 'green' : 'red'}">
                        $${data.unrealized_pnl.toFixed(2)}
                    </span>
                </div>
                <div class="stat">
                    <span class="label">Margin Used</span>
                    <span class="value">$${data.margin_used.toFixed(2)}</span>
                </div>
                <div class="stat">
                    <span class="label">Free Margin</span>
                    <span class="value">$${data.free_margin.toFixed(2)}</span>
                </div>
            `;
            document.getElementById('account-content').innerHTML = accountHtml;
            
            // Update Statistics
            const statsHtml = `
                <div class="stat">
                    <span class="label">Open Positions</span>
                    <span class="value">${data.open_positions}</span>
                </div>
                <div class="stat">
                    <span class="label">Signals Today</span>
                    <span class="value">${data.signals_today}</span>
                </div>
                <div class="stat">
                    <span class="label">Success Rate</span>
                    <span class="value green">${data.success_rate}%</span>
                </div>
                <div class="stat">
                    <span class="label">Avg Risk/Reward</span>
                    <span class="value">${data.avg_rr.toFixed(2)}</span>
                </div>
            `;
            document.getElementById('stats-content').innerHTML = statsHtml;
            
            // Update Positions
            if (data.positions.length > 0) {
                const positionsHtml = data.positions.map(pos => `
                    <div class="position-row">
                        <div class="position-header">
                            <span>${pos.symbol}</span>
                            <span class="${pos.pnl >= 0 ? 'green' : 'red'}">
                                ${pos.pnl >= 0 ? '+' : ''}$${pos.pnl.toFixed(2)}
                            </span>
                        </div>
                        <div class="position-details">
                            <span>Side: <strong>${pos.side}</strong></span>
                            <span>Size: <strong>${pos.size}</strong></span>
                            <span>Entry: <strong>$${pos.entry}</strong></span>
                            <span>Mark: <strong>$${pos.mark}</strong></span>
                        </div>
                    </div>
                `).join('');
                document.getElementById('positions-content').innerHTML = positionsHtml;
            } else {
                document.getElementById('positions-content').innerHTML = 
                    '<p style="color: #999; text-align: center;">No open positions</p>';
            }
            
            // Update Recent Signals
            if (data.recent_signals.length > 0) {
                const signalsHtml = data.recent_signals.map(sig => `
                    <div class="position-row">
                        <div class="position-header">
                            <span>${sig.symbol} ${sig.side}</span>
                            <span style="font-size: 0.85em; color: #999;">${sig.time_ago}</span>
                        </div>
                        <div class="position-details">
                            <span>Entries: <strong>${sig.entries}</strong></span>
                            <span>SL: <strong>$${sig.stop_loss}</strong></span>
                            <span>RR: <strong>${sig.risk_reward}</strong></span>
                            <span>Status: <strong>${sig.status}</strong></span>
                        </div>
                    </div>
                `).join('');
                document.getElementById('signals-content').innerHTML = signalsHtml;
            } else {
                document.getElementById('signals-content').innerHTML = 
                    '<p style="color: #999; text-align: center;">No recent signals</p>';
            }
            
            // Update Logs
            const logsHtml = data.recent_logs.map(log => {
                let className = 'log-line';
                if (log.includes('ERROR')) className += ' error';
                else if (log.includes('WARNING')) className += ' warning';
                else if (log.includes('INFO')) className += ' info';
                
                return `<div class="${className}">${log}</div>`;
            }).join('');
            document.getElementById('logs-content').innerHTML = logsHtml;
            
            // Auto-scroll logs to bottom
            const logViewer = document.getElementById('logs-content');
            logViewer.scrollTop = logViewer.scrollHeight;
        }
        
        function refreshData() {
            fetchData();
        }
        
        // Initial load
        fetchData();
        
        // Auto-refresh every 5 seconds
        setInterval(() => {
            if (autoRefresh) fetchData();
        }, 5000);
    </script>
</body>
</html>
"""

def get_bot_status():
    """Check if bot process is running"""
    try:
        import psutil
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmdline = process.info.get('cmdline', [])
            if cmdline and 'main.py' in ' '.join(cmdline):
                return True
    except:
        pass
    return False

def get_account_info():
    """Get account information from Binance"""
    if not binance_client:
        return {
            'balance': 0,
            'unrealized_pnl': 0,
            'margin_used': 0,
            'free_margin': 0
        }
    
    try:
        account = binance_client.account()
        return {
            'balance': float(account['totalWalletBalance']),
            'unrealized_pnl': float(account['unrealizedProfit']),
            'margin_used': float(account['totalInitialMargin']),
            'free_margin': float(account['availableBalance'])
        }
    except Exception as e:
        print(f"Error getting account info: {e}")
        return {
            'balance': 0,
            'unrealized_pnl': 0,
            'margin_used': 0,
            'free_margin': 0
        }

def get_positions():
    """Get open positions"""
    if not binance_client:
        return []
    
    try:
        account = binance_client.account()
        positions = []
        for pos in account['positions']:
            if float(pos['positionAmt']) != 0:
                positions.append({
                    'symbol': pos['symbol'],
                    'side': 'LONG' if float(pos['positionAmt']) > 0 else 'SHORT',
                    'size': abs(float(pos['positionAmt'])),
                    'entry': float(pos['entryPrice']),
                    'mark': float(pos['markPrice']),
                    'pnl': float(pos['unrealizedProfit'])
                })
        return positions
    except Exception as e:
        print(f"Error getting positions: {e}")
        return []

def get_recent_signals():
    """Get recent signals from database"""
    signals = []
    try:
        with sqlite3.connect(PERSISTENCE_FILE) as conn:
            cursor = conn.execute("""
                SELECT symbol, side, timestamp, processed_at 
                FROM processed_signals 
                ORDER BY processed_at DESC 
                LIMIT 10
            """)
            
            now = datetime.now()
            for row in cursor:
                timestamp = datetime.fromisoformat(row[2])
                time_diff = now - timestamp
                
                if time_diff.days > 0:
                    time_ago = f"{time_diff.days}d ago"
                elif time_diff.seconds > 3600:
                    time_ago = f"{time_diff.seconds // 3600}h ago"
                else:
                    time_ago = f"{time_diff.seconds // 60}m ago"
                
                signals.append({
                    'symbol': row[0],
                    'side': row[1],
                    'time_ago': time_ago,
                    'entries': 'N/A',
                    'stop_loss': 'N/A',
                    'risk_reward': 'N/A',
                    'status': 'PROCESSED'
                })
    except Exception as e:
        print(f"Error getting signals: {e}")
    
    return signals

def get_recent_logs(lines=50):
    """Get recent log lines"""
    logs = []
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                all_lines = f.readlines()
                logs = all_lines[-lines:] if len(all_lines) > lines else all_lines
                logs = [line.strip() for line in logs if line.strip()]
    except Exception as e:
        print(f"Error reading logs: {e}")
        logs = [f"Error reading log file: {e}"]
    
    return logs

def calculate_statistics():
    """Calculate trading statistics"""
    try:
        with sqlite3.connect(PERSISTENCE_FILE) as conn:
            # Signals today
            today_start = datetime.now().replace(hour=0, minute=0, second=0).isoformat()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM processed_signals WHERE processed_at > ?",
                (today_start,)
            )
            signals_today = cursor.fetchone()[0]
            
            # Get position count from state
            cursor = conn.execute(
                "SELECT value FROM bot_state WHERE key = 'position_count'"
            )
            row = cursor.fetchone()
            open_positions = json.loads(row[0]) if row else 0
            
            return {
                'signals_today': signals_today,
                'open_positions': open_positions,
                'success_rate': 75,  # Placeholder - implement actual calculation
                'avg_rr': 2.5  # Placeholder - implement actual calculation
            }
    except Exception as e:
        print(f"Error calculating stats: {e}")
        return {
            'signals_today': 0,
            'open_positions': 0,
            'success_rate': 0,
            'avg_rr': 0
        }

@app.route('/')
def dashboard():
    """Serve the dashboard HTML"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/data')
def api_data():
    """API endpoint for dashboard data"""
    account = get_account_info()
    stats = calculate_statistics()
    
    data = {
        'bot_running': get_bot_status(),
        'trading_mode': 'LIVE' if os.getenv('ENABLE_TRADING', 'false').lower() == 'true' else 'SIMULATION',
        'testnet': USE_TESTNET,
        'balance': account['balance'],
        'unrealized_pnl': account['unrealized_pnl'],
        'margin_used': account['margin_used'],
        'free_margin': account['free_margin'],
        'positions': get_positions(),
        'recent_signals': get_recent_signals(),
        'recent_logs': get_recent_logs(),
        'signals_today': stats['signals_today'],
        'open_positions': stats['open_positions'],
        'success_rate': stats['success_rate'],
        'avg_rr': stats['avg_rr']
    }
    
    return jsonify(data)

def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Run the dashboard server"""
    print(f"Starting dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_dashboard()
