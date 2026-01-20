# 🚀 Quick Start Guide

Get started with Fopper in under 5 minutes!

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SilFopma4h2/Fopper.git
   cd Fopper
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import flask, stable_baselines3; print('✓ All dependencies installed')"
   ```

## Running Your First Training Session

### Option 1: With Web Dashboard (Recommended)

Simply run:
```bash
python main.py
```

**What happens:**
- ✨ Web dashboard launches automatically at http://localhost:5000
- 🌐 Opens in your browser
- 🤖 Starts training PPO agent on EURUSD data
- 📊 Real-time monitoring:
  - Training progress bar
  - Live equity curve
  - Trade statistics
  - Position tracking
  - System logs

**Pro tip:** The dashboard auto-refreshes every 2 seconds. Leave it open to monitor your training!

### Option 2: Traditional Command Line

If you prefer command-line only:
```bash
python train_agent.py
```

This runs training without the dashboard but still saves all models and creates visualizations.

## What to Expect

**Training Duration:** ~10-30 minutes (600,000 timesteps)
- Checkpoint saved every 50,000 steps
- Progress logged to console
- TensorBoard logs available

**Outputs:**
- `model_eurusd_best.zip` - Best performing model
- `checkpoints/` - All checkpoint saves
- `tensorboard_log/` - Training metrics for TensorBoard

## Testing Your Trained Model

After training completes:
```bash
python test_agent.py
```

This will:
- Load your best model
- Run evaluation on test data
- Generate equity curve plot
- Save trade history to `trade_history_output.csv`

## Monitoring with TensorBoard (Optional)

While training is running, open another terminal:
```bash
tensorboard --logdir=./tensorboard_log/
```

Then visit http://localhost:6006 for detailed training metrics.

## Live Trading (Experimental)

⚠️ **Warning:** This is experimental. Use with caution and only with paper trading initially.

```python
from live_trading import start_live_trading
from dashboard import start_dashboard_thread

# Start dashboard
dashboard_thread = start_dashboard_thread()

# Start live trading
session = start_live_trading(
    model_path="model_eurusd_best",
    initial_balance=10000.0
)

# Monitor at http://localhost:5000
```

## Troubleshooting

**Problem:** Dashboard won't start
- **Solution:** Check if port 5000 is already in use. Change port in `main.py` line 293

**Problem:** Training crashes
- **Solution:** Ensure you have EURUSD data in the `data/` folder

**Problem:** Out of memory
- **Solution:** Reduce `total_timesteps` in `main.py` (line 280)

## Next Steps

1. 📖 Read the [full README](README.md) for detailed documentation
2. 🎓 Check [EXAMPLES.py](EXAMPLES.py) for code samples
3. ⚙️ Customize training parameters in `main.py`
4. 🔧 Modify environment settings in `trading_env.py`

## Need Help?

- 📝 See the [README](README.md) for comprehensive documentation
- 💡 Check [EXAMPLES.py](EXAMPLES.py) for usage patterns
- 🐛 Report issues on GitHub

---

**Happy Trading! 🎯**
