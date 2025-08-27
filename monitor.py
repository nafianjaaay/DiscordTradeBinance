#!/usr/bin/env python3
"""
Health monitoring script for the trading bot
Checks if bot is running and sends alerts if issues detected
"""

import os
import sys
import time
import json
import psutil
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
CHECK_INTERVAL = 300  # Check every 5 minutes
LOG_FILE = "logs/trading_bot.log"
ALERT_WEBHOOK = os.getenv("MONITOR_WEBHOOK_URL")  # Optional Discord webhook for alerts
MAX_LOG_AGE_MINUTES = 10  # Alert if no new logs for this duration

def check_process_running():
    """Check if the main bot process is running"""
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = process.info.get('cmdline', [])
            if cmdline and 'main.py' in ' '.join(cmdline):
                return True, process.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False, None

def check_log_activity():
    """Check if log file is being updated"""
    if not os.path.exists(LOG_FILE):
        return False, "Log file not found"
    
    # Get file modification time
    mod_time = os.path.getmtime(LOG_FILE)
    mod_datetime = datetime.fromtimestamp(mod_time)
    age_minutes = (datetime.now() - mod_datetime).total_seconds() / 60
    
    if age_minutes > MAX_LOG_AGE_MINUTES:
        return False, f"No logs for {age_minutes:.1f} minutes"
    
    return True, f"Last log {age_minutes:.1f} minutes ago"

def check_last_error():
    """Check for recent errors in log file"""
    if not os.path.exists(LOG_FILE):
        return None
    
    # Read last 100 lines
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()[-100:]
    
    errors = []
    for line in lines:
        if 'ERROR' in line or 'CRITICAL' in line:
            errors.append(line.strip())
    
    return errors[-5:] if errors else None  # Return last 5 errors

def send_alert(message, alert_type="warning"):
    """Send alert to Discord webhook"""
    if not ALERT_WEBHOOK:
        return
    
    colors = {
        "error": 15158332,    # Red
        "warning": 15844367,  # Yellow
        "info": 3447003       # Blue
    }
    
    payload = {
        "embeds": [{
            "title": f"🤖 Trading Bot Monitor Alert",
            "description": message,
            "color": colors.get(alert_type, colors["info"]),
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Bot Monitor"}
        }]
    }
    
    try:
        requests.post(ALERT_WEBHOOK, json=payload)
    except Exception as e:
        print(f"Failed to send alert: {e}")

def get_system_stats():
    """Get system resource usage"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }

def main():
    print("=" * 50)
    print("Trading Bot Monitor Started")
    print(f"Checking every {CHECK_INTERVAL} seconds")
    print("=" * 50)
    
    last_alert_time = {}
    
    while True:
        try:
            # Check if process is running
            is_running, pid = check_process_running()
            
            # Check log activity
            log_active, log_message = check_log_activity()
            
            # Check for errors
            recent_errors = check_last_error()
            
            # Get system stats
            stats = get_system_stats()
            
            # Generate status report
            status_parts = []
            alert_needed = False
            alert_type = "info"
            
            if is_running:
                status_parts.append(f"✅ Bot running (PID: {pid})")
            else:
                status_parts.append("❌ Bot not running!")
                alert_needed = True
                alert_type = "error"
            
            if log_active:
                status_parts.append(f"✅ Logs active ({log_message})")
            else:
                status_parts.append(f"⚠️ Log issue: {log_message}")
                if is_running:  # Only alert if bot should be running
                    alert_needed = True
                    alert_type = "warning"
            
            # Add system stats
            status_parts.append(f"CPU: {stats['cpu_percent']:.1f}%")
            status_parts.append(f"Memory: {stats['memory_percent']:.1f}%")
            status_parts.append(f"Disk: {stats['disk_percent']:.1f}%")
            
            # Add error summary if any
            if recent_errors:
                status_parts.append(f"⚠️ {len(recent_errors)} recent errors")
            
            # Print status
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Status Check:")
            for part in status_parts:
                print(f"  {part}")
            
            # Send alert if needed (with rate limiting)
            if alert_needed:
                alert_key = f"{alert_type}_{is_running}_{log_active}"
                last_alert = last_alert_time.get(alert_key, datetime.min)
                
                if datetime.now() - last_alert > timedelta(minutes=30):
                    alert_message = "\n".join(status_parts)
                    if recent_errors:
                        alert_message += "\n\n**Recent Errors:**\n```" + "\n".join(recent_errors[-3:]) + "```"
                    
                    send_alert(alert_message, alert_type)
                    last_alert_time[alert_key] = datetime.now()
                    print("  📢 Alert sent")
            
            # High resource usage warning
            if stats['cpu_percent'] > 80 or stats['memory_percent'] > 80:
                resource_key = "high_resource"
                last_alert = last_alert_time.get(resource_key, datetime.min)
                
                if datetime.now() - last_alert > timedelta(hours=1):
                    send_alert(
                        f"⚠️ High resource usage detected:\n"
                        f"CPU: {stats['cpu_percent']:.1f}%\n"
                        f"Memory: {stats['memory_percent']:.1f}%",
                        "warning"
                    )
                    last_alert_time[resource_key] = datetime.now()
            
        except KeyboardInterrupt:
            print("\nMonitor stopped by user")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
        
        # Wait for next check
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    # Check if psutil is installed
    try:
        import psutil
    except ImportError:
        print("Please install psutil: pip install psutil")
        sys.exit(1)
    
    main()
