"""
Web Dashboard for Fopper - Real-time Training and Trading Monitor
"""
import os
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# Shared state for dashboard data
dashboard_state = {
    "training_progress": {
        "current_timestep": 0,
        "total_timesteps": 0,
        "episode_count": 0,
        "mean_reward": 0.0,
        "best_reward": -float('inf'),
        "loss": 0.0,
    },
    "equity_curve": [],
    "trade_history": [],
    "current_position": {
        "type": "FLAT",  # FLAT, LONG, SHORT
        "entry_price": None,
        "sl_price": None,
        "tp_price": None,
        "unrealized_pnl": 0.0,
        "time_in_trade": 0,
    },
    "metrics": {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
    },
    "status": "IDLE",  # IDLE, TRAINING, LIVE_TRADING
    "logs": [],
}

dashboard_lock = threading.Lock()


def add_log(message, level="INFO"):
    """Add a log entry to the dashboard"""
    with dashboard_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        dashboard_state["logs"].append(log_entry)
        # Keep only last 100 logs
        if len(dashboard_state["logs"]) > 100:
            dashboard_state["logs"] = dashboard_state["logs"][-100:]


def update_training_progress(timestep, total_timesteps, episode_count, mean_reward, loss=None):
    """Update training progress metrics"""
    with dashboard_lock:
        dashboard_state["training_progress"]["current_timestep"] = timestep
        dashboard_state["training_progress"]["total_timesteps"] = total_timesteps
        dashboard_state["training_progress"]["episode_count"] = episode_count
        dashboard_state["training_progress"]["mean_reward"] = mean_reward
        if mean_reward > dashboard_state["training_progress"]["best_reward"]:
            dashboard_state["training_progress"]["best_reward"] = mean_reward
        if loss is not None:
            dashboard_state["training_progress"]["loss"] = loss


def update_equity_curve(equity_value):
    """Add equity point to the curve"""
    with dashboard_lock:
        dashboard_state["equity_curve"].append({
            "timestamp": datetime.now().isoformat(),
            "equity": equity_value
        })
        # Keep last 10000 points
        if len(dashboard_state["equity_curve"]) > 10000:
            dashboard_state["equity_curve"] = dashboard_state["equity_curve"][-10000:]


def update_position(position_type, entry_price=None, sl_price=None, tp_price=None, 
                   unrealized_pnl=0.0, time_in_trade=0):
    """Update current position state"""
    with dashboard_lock:
        dashboard_state["current_position"]["type"] = position_type
        dashboard_state["current_position"]["entry_price"] = entry_price
        dashboard_state["current_position"]["sl_price"] = sl_price
        dashboard_state["current_position"]["tp_price"] = tp_price
        dashboard_state["current_position"]["unrealized_pnl"] = unrealized_pnl
        dashboard_state["current_position"]["time_in_trade"] = time_in_trade


def add_trade(trade_info):
    """Add a closed trade to history"""
    with dashboard_lock:
        dashboard_state["trade_history"].append(trade_info)
        # Keep last 1000 trades
        if len(dashboard_state["trade_history"]) > 1000:
            dashboard_state["trade_history"] = dashboard_state["trade_history"][-1000:]
        
        # Update metrics
        calculate_metrics()


def calculate_metrics():
    """Calculate trading metrics from trade history"""
    trades = dashboard_state["trade_history"]
    if not trades:
        return
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.get("net_pips", 0) > 0)
    losing_trades = sum(1 for t in trades if t.get("net_pips", 0) <= 0)
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    gross_profit = sum(t.get("net_pips", 0) for t in trades if t.get("net_pips", 0) > 0)
    gross_loss = abs(sum(t.get("net_pips", 0) for t in trades if t.get("net_pips", 0) < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    
    # Calculate max drawdown from equity curve
    equity_values = [point["equity"] for point in dashboard_state["equity_curve"]]
    if equity_values:
        peak = equity_values[0]
        max_dd = 0.0
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = ((peak - equity) / peak * 100) if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
    else:
        max_dd = 0.0
    
    dashboard_state["metrics"]["total_trades"] = total_trades
    dashboard_state["metrics"]["winning_trades"] = winning_trades
    dashboard_state["metrics"]["losing_trades"] = losing_trades
    dashboard_state["metrics"]["win_rate"] = win_rate
    dashboard_state["metrics"]["profit_factor"] = profit_factor
    dashboard_state["metrics"]["max_drawdown"] = max_dd


def set_status(status):
    """Set dashboard status"""
    with dashboard_lock:
        dashboard_state["status"] = status
        add_log(f"Status changed to: {status}")


# ===== Flask Routes =====

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/state')
def get_state():
    """Get current dashboard state"""
    with dashboard_lock:
        return jsonify(dashboard_state)


@app.route('/api/equity_plot')
def get_equity_plot():
    """Generate and return equity curve plot as base64 image"""
    with dashboard_lock:
        equity_data = dashboard_state["equity_curve"]
    
    if not equity_data:
        return jsonify({"image": None, "message": "No data available"})
    
    try:
        equity_values = [point["equity"] for point in equity_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(equity_values, linewidth=2, color='#2E86AB')
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Annotate final equity
        if equity_values:
            final_equity = equity_values[-1]
            ax.annotate(f'${final_equity:,.2f}', 
                       xy=(len(equity_values)-1, final_equity),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({"image": f"data:image/png;base64,{img_base64}"})
    except Exception as e:
        return jsonify({"image": None, "error": str(e)})


@app.route('/api/logs')
def get_logs():
    """Get recent logs"""
    with dashboard_lock:
        return jsonify(dashboard_state["logs"][-50:])


@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    """Clear all logs"""
    with dashboard_lock:
        dashboard_state["logs"] = []
    return jsonify({"status": "success"})


def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask dashboard"""
    add_log(f"Starting dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def start_dashboard_thread(host='0.0.0.0', port=5000):
    """Start dashboard in a separate thread"""
    dashboard_thread = threading.Thread(
        target=run_dashboard,
        args=(host, port, False),
        daemon=True
    )
    dashboard_thread.start()
    add_log("Dashboard thread started")
    return dashboard_thread
