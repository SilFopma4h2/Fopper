#!/usr/bin/env python3
"""
Main entry point for Fopper - Forex Trading RL Agent
Starts training with live web dashboard monitoring
"""
import os
import sys
import time
import webbrowser
import threading
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from dashboard import (
    start_dashboard_thread, 
    add_log, 
    set_status, 
    update_training_progress,
    update_equity_curve,
    update_position,
    add_trade
)


class DashboardCallback(BaseCallback):
    """
    Custom callback to update the dashboard with training progress
    """
    def __init__(self, eval_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Update dashboard with current training progress
        if self.num_timesteps % 100 == 0:  # Update every 100 steps
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            
            # Get current loss from logger if available
            loss = None
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                loss = self.model.logger.name_to_value.get('train/loss', None)
            
            update_training_progress(
                timestep=self.num_timesteps,
                total_timesteps=self.model._total_timesteps,
                episode_count=self.episode_count,
                mean_reward=mean_reward,
                loss=loss
            )
        
        # Track rewards and update equity
        for info in self.locals.get("infos", []):
            if "equity_usd" in info:
                equity = info["equity_usd"]
                update_equity_curve(equity)
                
                # Update position info
                position = info.get("position", 0)
                position_type = "FLAT" if position == 0 else ("LONG" if position == 1 else "SHORT")
                
                # Get trade info if position changed
                last_trade_info = info.get("last_trade_info")
                if last_trade_info and last_trade_info.get("event") == "CLOSE":
                    add_trade(last_trade_info)
                    add_log(
                        f"Trade closed: {last_trade_info.get('reason')} | "
                        f"PnL: {last_trade_info.get('net_pips', 0):.2f} pips | "
                        f"Equity: ${last_trade_info.get('equity_usd', 0):.2f}",
                        level="SUCCESS" if last_trade_info.get('net_pips', 0) > 0 else "WARNING"
                    )
                
                # Update current position on dashboard
                time_in_trade = info.get("time_in_trade", 0)
                if position != 0:
                    # Get env to extract position details
                    try:
                        env = self.training_env.envs[0]
                        update_position(
                            position_type=position_type,
                            entry_price=env.entry_price,
                            sl_price=env.sl_price,
                            tp_price=env.tp_price,
                            unrealized_pnl=env._compute_unrealized_pips(),
                            time_in_trade=time_in_trade
                        )
                    except:
                        update_position(position_type=position_type)
                else:
                    update_position(position_type="FLAT")
        
        # Evaluate periodically
        if self.num_timesteps % self.eval_freq == 0:
            add_log(f"Evaluation at timestep {self.num_timesteps}")
            try:
                obs = self.eval_env.reset()
                total_reward = 0
                done = False
                steps = 0
                
                while not done and steps < 1000:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_out = self.eval_env.step(action)
                    
                    if len(step_out) == 4:
                        obs, reward, done, info = step_out
                        done = bool(done[0] if isinstance(done, (list, np.ndarray)) else done)
                    else:
                        obs, reward, terminated, truncated, info = step_out
                        done = bool((terminated[0] if isinstance(terminated, (list, np.ndarray)) else terminated) or 
                                  (truncated[0] if isinstance(truncated, (list, np.ndarray)) else truncated))
                    
                    total_reward += float(reward[0] if isinstance(reward, (list, np.ndarray)) else reward)
                    steps += 1
                
                add_log(f"Eval episode reward: {total_reward:.2f} (steps: {steps})", level="INFO")
            except Exception as e:
                add_log(f"Evaluation failed: {e}", level="WARNING")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        self.episode_count += 1


def train_agent(
    data_file="data/EURUSD_Hourly_Ask_2015.12.01_2025.12.16.csv",
    total_timesteps=600000,
    checkpoint_freq=50000
):
    """
    Main training function with dashboard integration
    """
    add_log("Starting Fopper training system", level="SUCCESS")
    set_status("TRAINING")
    
    # Load and preprocess data
    add_log(f"Loading data from {data_file}")
    df, feature_cols = load_and_preprocess_data(data_file)
    add_log(f"Data loaded: {len(df)} bars", level="SUCCESS")
    
    # Time split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    add_log(f"Training bars: {len(train_df)}")
    add_log(f"Testing bars: {len(test_df)}")
    
    # Environment parameters
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    WIN = 30
    
    # Create environment factories
    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=True,
            min_episode_steps=1000,
            episode_max_steps=2000,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0
        )
    
    def make_eval_env():
        return ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0
        )
    
    train_vec_env = DummyVecEnv([make_train_env])
    eval_vec_env = DummyVecEnv([make_eval_env])
    
    add_log("Environments created", level="SUCCESS")
    
    # Create model
    add_log("Initializing PPO model")
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/"
    )
    
    # Setup checkpoints
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="ppo_eurusd"
    )
    
    dashboard_callback = DashboardCallback(eval_env=eval_vec_env, eval_freq=10000)
    
    callback_list = CallbackList([checkpoint_callback, dashboard_callback])
    
    # Train
    add_log(f"Starting training for {total_timesteps} timesteps", level="SUCCESS")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback_list)
        add_log("Training completed!", level="SUCCESS")
    except KeyboardInterrupt:
        add_log("Training interrupted by user", level="WARNING")
    except Exception as e:
        add_log(f"Training error: {e}", level="ERROR")
        raise
    
    # Save final model
    model.save("model_eurusd_final")
    add_log("Final model saved as model_eurusd_final", level="SUCCESS")
    
    # Evaluate best checkpoint
    add_log("Evaluating checkpoints to select best model")
    best_equity = -np.inf
    best_path = None
    
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip") and f.startswith("ppo_eurusd")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )
    
    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            m = PPO.load(ck_path, env=eval_vec_env)
            obs = eval_vec_env.reset()
            equity_curve = []
            
            while True:
                action, _ = m.predict(obs, deterministic=True)
                step_out = eval_vec_env.step(action)
                
                if len(step_out) == 4:
                    obs, rewards, dones, infos = step_out
                    done = bool(dones[0])
                else:
                    obs, rewards, terminated, truncated, infos = step_out
                    done = bool(terminated[0] or truncated[0])
                
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                eq = info.get("equity_usd", eval_vec_env.get_attr("equity_usd")[0])
                equity_curve.append(eq)
                
                if done:
                    break
            
            final_equity = float(equity_curve[-1])
            add_log(f"Checkpoint {ck} -> final equity: ${final_equity:.2f}")
            
            if final_equity > best_equity:
                best_equity = final_equity
                best_path = ck_path
        except Exception as e:
            add_log(f"Could not evaluate checkpoint {ck}: {e}", level="WARNING")
    
    # Save best model
    if best_path:
        best_model = PPO.load(best_path, env=train_vec_env)
        best_model.save("model_eurusd_best")
        add_log(f"Best model saved: model_eurusd_best (equity: ${best_equity:.2f})", level="SUCCESS")
    else:
        model.save("model_eurusd_best")
        add_log("Using final model as best model", level="SUCCESS")
    
    set_status("IDLE")
    add_log("Training session complete. Dashboard will continue running.", level="SUCCESS")


def main():
    """
    Main entry point - starts dashboard and training
    """
    print("=" * 70)
    print("🚀 Fopper - Forex Trading RL Agent")
    print("=" * 70)
    print()
    
    # Start dashboard in background thread
    print("Starting web dashboard...")
    dashboard_thread = start_dashboard_thread(host='0.0.0.0', port=5000)
    
    # Give dashboard time to start
    time.sleep(2)
    
    # Open browser
    dashboard_url = "http://localhost:5000"
    print(f"\n✨ Dashboard available at: {dashboard_url}")
    print("Opening dashboard in browser...")
    
    try:
        webbrowser.open(dashboard_url)
    except:
        pass
    
    print("\n" + "=" * 70)
    print("Training will begin in 3 seconds...")
    print("Monitor progress on the web dashboard")
    print("Press Ctrl+C to stop training")
    print("=" * 70 + "\n")
    
    time.sleep(3)
    
    # Start training
    try:
        train_agent(
            data_file="data/EURUSD_Hourly_Ask_2015.12.01_2025.12.16.csv",
            total_timesteps=600000,
            checkpoint_freq=50000
        )
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user")
        set_status("IDLE")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        add_log(f"Fatal error: {e}", level="ERROR")
        set_status("IDLE")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Dashboard still running at: {dashboard_url}")
    print("Press Ctrl+C to exit")
    print("=" * 70 + "\n")
    
    # Keep main thread alive to keep dashboard running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
