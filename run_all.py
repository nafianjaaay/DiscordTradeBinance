#!/usr/bin/env python3
"""
Launcher script to run both bot and dashboard
"""

import subprocess
import sys
import time
import signal
import os
from threading import Thread

processes = []

def signal_handler(sig, frame):
    """Handle shutdown gracefully"""
    print("\n[LAUNCHER] Shutting down services...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
            time.sleep(1)
            if p.poll() is None:
                p.kill()
    sys.exit(0)

def run_bot():
    """Start the trading bot"""
    print("[LAUNCHER] Starting trading bot...")
    p = subprocess.Popen([sys.executable, "main.py"])
    processes.append(p)
    p.wait()

def run_dashboard():
    """Start the web dashboard"""
    print("[LAUNCHER] Starting web dashboard...")
    time.sleep(2)  # Let bot initialize first
    p = subprocess.Popen([sys.executable, "dashboard.py"])
    processes.append(p)
    p.wait()

def main():
    """Main launcher"""
    print("=" * 50)
    print("Trading Bot & Dashboard Launcher")
    print("=" * 50)
    print("\nStarting services...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop all services\n")
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start bot in thread
    bot_thread = Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Start dashboard in thread
    dashboard_thread = Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
            # Check if processes are still running
            for p in processes:
                if p.poll() is not None:
                    print(f"[LAUNCHER] Process {p.pid} has stopped")
                    # Optionally restart the process
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
