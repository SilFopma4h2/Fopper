#!/usr/bin/env python3
"""
Demo script showing how to use the Fopper dashboard and live trading features
"""

# Example 1: Using main.py for training with dashboard
print("""
Example 1: Training with Web Dashboard
=======================================

python main.py

This will:
1. Start the web dashboard on http://localhost:5000
2. Automatically open it in your browser
3. Begin training your RL agent
4. Display real-time metrics:
   - Training progress
   - Equity curve
   - Trade statistics
   - System logs

The dashboard auto-refreshes every 2 seconds.
""")

# Example 2: Programmatic dashboard usage
print("""
Example 2: Programmatic Dashboard Usage
========================================

from dashboard import (
    start_dashboard_thread, 
    add_log, 
    set_status,
    update_training_progress,
    update_equity_curve
)

# Start dashboard
dashboard_thread = start_dashboard_thread(host='0.0.0.0', port=5000)

# Update dashboard during your custom training loop
add_log("Starting custom training", level="INFO")
set_status("TRAINING")

for step in range(1000):
    # Your training code here
    
    # Update dashboard
    update_training_progress(
        timestep=step,
        total_timesteps=1000,
        episode_count=step // 100,
        mean_reward=10.5 + step * 0.01
    )
    update_equity_curve(10000 + step * 10)

set_status("IDLE")
add_log("Training complete", level="SUCCESS")
""")

# Example 3: Live trading
print("""
Example 3: Live Trading (Experimental)
=======================================

from live_trading import start_live_trading
from dashboard import start_dashboard_thread

# Start dashboard first
dashboard_thread = start_dashboard_thread()

# Start live trading session
session = start_live_trading(
    model_path="model_eurusd_best",
    initial_balance=10000.0
)

# Session runs in background
# Monitor via dashboard at http://localhost:5000

# Stop when done
# session.stop()
""")

# Example 4: Dashboard API endpoints
print("""
Example 4: Dashboard API Endpoints
===================================

Once the dashboard is running, you can access these endpoints:

- http://localhost:5000/              Main dashboard UI
- http://localhost:5000/api/state     Get current dashboard state (JSON)
- http://localhost:5000/api/equity_plot   Get equity curve plot
- http://localhost:5000/api/logs      Get recent logs

You can integrate these endpoints with your own monitoring tools.
""")

print("""
For more information, see:
- main.py - Complete training with dashboard example
- dashboard.py - Dashboard implementation and functions
- live_trading.py - Live trading session management
- README.md - Full documentation
""")
