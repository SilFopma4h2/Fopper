"""
Live Trading Functions for Fopper
Provides real-time trading capabilities with live data feeds
"""
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

from stable_baselines3 import PPO
from dashboard import (
    add_log,
    set_status,
    update_equity_curve,
    update_position,
    add_trade
)


class LiveTradingSession:
    """
    Manages a live trading session with real-time data and model predictions
    """
    def __init__(
        self,
        model_path: str,
        initial_balance: float = 10000.0,
        pip_value: float = 0.0001,
        lot_size: float = 100000.0
    ):
        """
        Initialize live trading session
        
        Args:
            model_path: Path to trained model
            initial_balance: Starting balance in USD
            pip_value: Value of one pip (default 0.0001 for EURUSD)
            lot_size: Standard lot size
        """
        self.model = PPO.load(model_path)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.pip_value = pip_value
        self.lot_size = lot_size
        self.usd_per_pip = pip_value * lot_size
        
        # Position tracking
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        
        # Trading state
        self.is_running = False
        self.thread = None
        
        # Data buffer for observations
        self.price_history = []
        self.window_size = 30
        
        add_log("Live trading session initialized", level="SUCCESS")
    
    def start(self, data_feed_func=None, update_interval=1.0):
        """
        Start live trading session
        
        Args:
            data_feed_func: Function that returns latest price data (OHLCV)
            update_interval: Seconds between updates
        """
        if self.is_running:
            add_log("Live trading already running", level="WARNING")
            return
        
        self.is_running = True
        set_status("LIVE_TRADING")
        add_log("Starting live trading session", level="SUCCESS")
        
        # Start trading loop in background thread
        self.thread = threading.Thread(
            target=self._trading_loop,
            args=(data_feed_func, update_interval),
            daemon=True
        )
        self.thread.start()
    
    def stop(self):
        """Stop live trading session"""
        if not self.is_running:
            return
        
        add_log("Stopping live trading session", level="WARNING")
        self.is_running = False
        
        # Close any open positions
        if self.position != 0:
            self._close_position("SESSION_STOPPED")
        
        set_status("IDLE")
        add_log("Live trading session stopped", level="SUCCESS")
    
    def _trading_loop(self, data_feed_func, update_interval):
        """
        Main trading loop - runs in background thread
        """
        while self.is_running:
            try:
                # Get latest price data
                if data_feed_func:
                    price_data = data_feed_func()
                else:
                    # Simulated data for testing
                    price_data = self._simulate_price_tick()
                
                # Process tick
                self._process_tick(price_data)
                
                # Update dashboard
                self._update_dashboard()
                
                # Sleep until next update
                time.sleep(update_interval)
                
            except Exception as e:
                add_log(f"Error in trading loop: {e}", level="ERROR")
                time.sleep(update_interval)
    
    def _simulate_price_tick(self) -> Dict[str, float]:
        """
        Simulate a price tick for testing
        Returns OHLCV data
        """
        # Simple random walk for testing
        if not hasattr(self, '_last_price'):
            self._last_price = 1.05000
        
        change = np.random.randn() * 0.0001
        self._last_price += change
        
        return {
            'timestamp': datetime.now(),
            'open': self._last_price - abs(np.random.randn() * 0.00005),
            'high': self._last_price + abs(np.random.randn() * 0.00010),
            'low': self._last_price - abs(np.random.randn() * 0.00010),
            'close': self._last_price,
            'volume': np.random.randint(1000, 10000)
        }
    
    def _process_tick(self, price_data: Dict[str, float]):
        """
        Process a single price tick
        
        Args:
            price_data: Dictionary with OHLCV data
        """
        current_price = price_data['close']
        
        # Add to price history
        self.price_history.append(price_data)
        if len(self.price_history) > self.window_size:
            self.price_history = self.price_history[-self.window_size:]
        
        # Check if we have enough data
        if len(self.price_history) < self.window_size:
            return
        
        # Check SL/TP if in position
        if self.position != 0:
            self.time_in_trade += 1
            
            high = price_data['high']
            low = price_data['low']
            
            if self.position == 1:  # Long
                if low <= self.sl_price:
                    self._close_position("SL_HIT", self.sl_price)
                    return
                elif high >= self.tp_price:
                    self._close_position("TP_HIT", self.tp_price)
                    return
            else:  # Short
                if high >= self.sl_price:
                    self._close_position("SL_HIT", self.sl_price)
                    return
                elif low <= self.tp_price:
                    self._close_position("TP_HIT", self.tp_price)
                    return
        
        # Get model prediction (simplified - would need proper feature engineering)
        # For now, just demonstrate the structure
        # In production, you'd create proper observations from price_history
        
        # Create observation (simplified version)
        obs = self._create_observation()
        
        if obs is not None:
            # Get model action
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Execute action (simplified)
            # In real implementation, this would map to your action space
            self._execute_action(int(action), current_price)
    
    def _create_observation(self) -> Optional[np.ndarray]:
        """
        Create observation from price history
        This is a simplified version - would need full technical indicators
        """
        # This would need to match your training observation space
        # For now, return None to avoid errors
        # In production, compute technical indicators from price_history
        return None
    
    def _execute_action(self, action: int, current_price: float):
        """
        Execute trading action
        
        Args:
            action: Action index from model
            current_price: Current market price
        """
        # Simplified action execution
        # In production, this would map to your full action space
        
        if action == 0:  # HOLD
            pass
        elif action == 1:  # CLOSE
            if self.position != 0:
                self._close_position("MANUAL_CLOSE", current_price)
        else:
            # OPEN position (simplified)
            # In production, decode direction, SL, TP from action
            pass
    
    def _close_position(self, reason: str, exit_price: float):
        """
        Close current position
        
        Args:
            reason: Reason for closing
            exit_price: Exit price
        """
        if self.position == 0:
            return
        
        # Calculate PnL
        if self.position == 1:  # Long
            pnl_price = exit_price - self.entry_price
        else:  # Short
            pnl_price = self.entry_price - exit_price
        
        pnl_pips = pnl_price / self.pip_value
        pnl_usd = pnl_pips * self.usd_per_pip
        
        self.balance += pnl_usd
        
        # Log trade
        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "position": "LONG" if self.position == 1 else "SHORT",
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(pnl_pips),
            "net_pips": float(pnl_pips),
            "pnl_usd": float(pnl_usd),
            "equity_usd": float(self.balance),
            "time_in_trade": int(self.time_in_trade)
        }
        
        add_trade(trade_info)
        add_log(
            f"Position closed: {reason} | PnL: {pnl_pips:.2f} pips (${pnl_usd:.2f}) | Balance: ${self.balance:.2f}",
            level="SUCCESS" if pnl_pips > 0 else "WARNING"
        )
        
        # Reset position
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
    
    def _update_dashboard(self):
        """Update dashboard with current state"""
        # Update equity
        update_equity_curve(self.balance)
        
        # Update position
        if self.position != 0:
            position_type = "LONG" if self.position == 1 else "SHORT"
            
            # Calculate unrealized PnL
            if hasattr(self, '_last_price'):
                current_price = self._last_price
                if self.position == 1:
                    unrealized_pips = (current_price - self.entry_price) / self.pip_value
                else:
                    unrealized_pips = (self.entry_price - current_price) / self.pip_value
            else:
                unrealized_pips = 0.0
            
            update_position(
                position_type=position_type,
                entry_price=self.entry_price,
                sl_price=self.sl_price,
                tp_price=self.tp_price,
                unrealized_pnl=unrealized_pips,
                time_in_trade=self.time_in_trade
            )
        else:
            update_position(position_type="FLAT")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        return {
            "is_running": self.is_running,
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "time_in_trade": self.time_in_trade,
            "pnl_usd": self.balance - self.initial_balance
        }


def start_live_trading(model_path: str, initial_balance: float = 10000.0) -> LiveTradingSession:
    """
    Start a live trading session
    
    Args:
        model_path: Path to trained model file
        initial_balance: Starting balance in USD
        
    Returns:
        LiveTradingSession instance
    """
    session = LiveTradingSession(model_path, initial_balance)
    session.start()
    return session
