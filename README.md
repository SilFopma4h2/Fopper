# 🚀 Fopper - Forex Trading RL Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A reinforcement learning-based Forex trading agent built with Stable-Baselines3 and custom Gymnasium environment. Fopper uses PPO (Proximal Policy Optimization) to learn trading strategies on EURUSD historical data with realistic market conditions including spreads, slippage, and stop-loss/take-profit mechanics.

## ✨ Features

- 🤖 **RL-Powered Trading**: Uses PPO algorithm from Stable-Baselines3
- 🌐 **Web Dashboard**: Real-time monitoring with Flask-based dashboard
  - Live training progress and metrics
  - Interactive equity curves
  - Position tracking and PnL
  - Trade history and statistics
  - Auto-refreshing interface
- 📊 **Custom Gym Environment**: Realistic Forex trading simulation with:
  - Position persistence (long/short/flat)
  - Configurable stop-loss and take-profit levels
  - Market friction modeling (spread, commission, slippage)
  - Intrabar SL/TP hit detection
- 📈 **Technical Indicators**: Powered by pandas-ta (RSI, ATR, Moving Averages)
- 🎯 **Smart Reward Shaping**: Combines realized PnL with optional unrealized PnL tracking
- 🔄 **Train/Test Split**: Proper time-series validation with in-sample/out-of-sample evaluation
- 💾 **Model Checkpointing**: Automatic model saving and best model selection
- 📉 **Equity Curve Visualization**: Track agent performance over time
- 🔴 **Live Trading Support**: Framework for real-time trading (experimental)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/SilFopma4h2/Fopper.git
cd Fopper
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your data**
   - Place your EURUSD CSV files in the `data/` directory
   - Expected format: `Time (EET), Open, High, Low, Close, Volume`

## 🚀 Usage

### Quick Start with Dashboard (Recommended)

The easiest way to train and monitor your agent is using the main.py script with integrated web dashboard:

```bash
python main.py
```

This will:
- **Automatically start a web dashboard** on http://localhost:5000
- Open the dashboard in your browser
- Begin training with real-time monitoring
- Display live metrics including:
  - Training progress and timesteps
  - Real-time equity curve
  - Current positions and PnL
  - Trade history and statistics
  - System logs

The dashboard updates every 2 seconds and provides a comprehensive view of your training session.

### Training Without Dashboard

For traditional command-line training:

```bash
python train_agent.py
```

This will:
- Load and preprocess EURUSD data
- Split into 80% training / 20% testing
- Train a PPO agent for 600,000 timesteps
- Save checkpoints every 50,000 steps
- Evaluate all checkpoints on out-of-sample data
- Select and save the best model as `model_eurusd_best.zip`
- Display equity curves for both in-sample and out-of-sample performance

### Testing a Trained Agent

```bash
python test_agent.py
```

This will:
- Load the best model
- Run evaluation on test data
- Save trade history to `trade_history_output.csv`
- Display the equity curve

### Live Trading (Experimental)

For live trading with a trained model:

```python
from live_trading import start_live_trading

# Start live trading session
session = start_live_trading(
    model_path="model_eurusd_best",
    initial_balance=10000.0
)

# Session runs in background with dashboard updates
# Stop with: session.stop()
```

**Note**: Live trading is experimental and requires integration with a live data feed.

## 📁 Project Structure

```
Fopper/
├── main.py                 # 🆕 Main entry point with integrated dashboard
├── train_agent.py          # Traditional training script
├── test_agent.py           # Model evaluation script
├── trading_env.py          # Custom Gymnasium trading environment
├── indicators.py           # Technical indicator preprocessing
├── dashboard.py            # 🆕 Web dashboard backend (Flask)
├── live_trading.py         # 🆕 Live trading functionality
├── requirements.txt        # Python dependencies
├── templates/              # 🆕 Dashboard HTML templates
│   └── dashboard.html      # Main dashboard interface
├── data/                   # Historical EURUSD data (CSV files)
├── checkpoints/            # Model checkpoints (auto-generated)
├── tensorboard_log/        # TensorBoard logs (auto-generated)
└── README.md              # This file
```

## 🎮 Environment Details

### Action Space
- **0**: HOLD - Do nothing
- **1**: CLOSE - Close current position
- **2+**: OPEN - Open position with specific direction, stop-loss, and take-profit

### Observation Space
- Rolling window of technical indicators (default: 30 timesteps)
- Position state features (position type, time in trade, unrealized PnL)

### Rewards
- Realized PnL in pips (minus costs) on trade closes
- Optional shaping based on unrealized PnL changes
- Configurable penalties for overtrading and holding duration

## 🔧 Customization

### Modify Trading Parameters

Edit `train_agent.py` to customize:
```python
SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]  # Stop-loss options (pips)
TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]  # Take-profit options (pips)
WIN = 30                                      # Observation window size
spread_pips = 1.0                             # Bid-ask spread
```

### Adjust Training Hyperparameters

```python
model = PPO(
    policy="MlpPolicy",
    env=train_vec_env,
    learning_rate=3e-4,        # Add custom learning rate
    n_steps=2048,              # Steps per update
    batch_size=64,             # Batch size
    verbose=1,
    tensorboard_log="./tensorboard_log/"
)
```

## 📊 Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir=./tensorboard_log/
```

Then open http://localhost:6006 in your browser.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** Trading Forex carries significant risk of loss. This agent should NOT be used for live trading without extensive additional testing, risk management, and professional advice. Past performance does not guarantee future results.

## 🙏 Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the RL framework
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for the environment API
- [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical analysis indicators

---

**Built with ❤️ for quantitative trading research**
