"""
Training Callbacks Module for Arabic Sign Language Recognition
Provides various callbacks for training monitoring, early stopping, and learning rate scheduling
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod


class Callback(ABC):
    """Base class for all training callbacks"""
    
    def __init__(self):
        self.trainer = None
    
    def set_trainer(self, trainer):
        """Set the trainer instance"""
        self.trainer = trainer
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training"""
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each batch"""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of each batch"""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to prevent overfitting
    Monitors a metric and stops training when it stops improving
    """
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10, 
                 min_delta: float = 0.001, mode: str = 'min', 
                 restore_best_weights: bool = True, verbose: bool = True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best = float('inf')
        else:  # mode == 'max'
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best = float('-inf')
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Reset early stopping state"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if self.mode == 'min':
            self.best = float('inf')
        else:
            self.best = float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check if training should be stopped"""
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose:
                print(f"Early stopping conditioned on metric `{self.monitor}` which is not available. Training continues.")
            return
        
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights and self.trainer is not None:
                self.best_weights = {k: v.clone() for k, v in self.trainer.model.state_dict().items()}
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                
                # Restore best weights
                if self.restore_best_weights and self.best_weights is not None and self.trainer is not None:
                    self.trainer.model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("Restored best weights")
                
                # Stop training
                if self.trainer is not None:
                    self.trainer.stop_training = True
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Print early stopping summary"""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Training stopped early at epoch {self.stopped_epoch + 1}")
            print(f"Best {self.monitor}: {self.best:.4f}")


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when a metric has stopped improving
    """
    
    def __init__(self, monitor: str = 'val_loss', factor: float = 0.5, 
                 patience: int = 5, min_delta: float = 0.001, 
                 mode: str = 'min', min_lr: float = 1e-7, verbose: bool = True):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.wait = 0
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best = float('inf')
        else:  # mode == 'max'
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best = float('-inf')
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Reset state"""
        self.wait = 0
        if self.mode == 'min':
            self.best = float('inf')
        else:
            self.best = float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check if learning rate should be reduced"""
        if logs is None or self.trainer is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                # Get current learning rate
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * self.factor, self.min_lr)
                
                if new_lr < current_lr:
                    # Update learning rate
                    for param_group in self.trainer.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    if self.verbose:
                        print(f"Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
                    
                    self.wait = 0


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training
    """
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 save_best_only: bool = True, mode: str = 'min', 
                 save_weights_only: bool = False, verbose: bool = True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best
            self.best = float('inf')
        else:  # mode == 'max'
            self.monitor_op = lambda current, best: current > best
            self.best = float('-inf')
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Reset state"""
        if self.mode == 'min':
            self.best = float('inf')
        else:
            self.best = float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save checkpoint if conditions are met"""
        if logs is None or self.trainer is None:
            return
        
        # Format filepath with epoch and metrics
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                return
            
            if self.monitor_op(current, self.best):
                self.best = current
                self._save_checkpoint(filepath, epoch, logs)
        else:
            self._save_checkpoint(filepath, epoch, logs)
    
    def _save_checkpoint(self, filepath: str, epoch: int, logs: Dict):
        """Save the actual checkpoint"""
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if self.save_weights_only:
            # Save only model weights
            torch.save(self.trainer.model.state_dict(), filepath)
        else:
            # Save full checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.trainer.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'metrics': logs,
                'model_config': getattr(self.trainer.model, 'config', None)
            }
            
            if hasattr(self.trainer, 'scheduler') and self.trainer.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.trainer.scheduler.state_dict()
            
            torch.save(checkpoint, filepath)
        
        if self.verbose:
            print(f"Checkpoint saved: {filepath}")


class CSVLogger(Callback):
    """
    Log training metrics to a CSV file
    """
    
    def __init__(self, filename: str, separator: str = ',', append: bool = False):
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        self.keys = None
        self.file = None
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Open CSV file for writing"""
        mode = 'a' if self.append else 'w'
        
        # Create directory if it doesn't exist
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        
        self.file = open(self.filename, mode, newline='')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Write metrics to CSV"""
        if logs is None or self.file is None:
            return
        
        # Add epoch to logs
        logs = dict(logs)
        logs['epoch'] = epoch + 1
        
        if self.keys is None:
            # First epoch - write header
            self.keys = sorted(logs.keys())
            self.file.write(self.separator.join(self.keys) + '\n')
        
        # Write values
        values = [str(logs.get(key, '')) for key in self.keys]
        self.file.write(self.separator.join(values) + '\n')
        self.file.flush()
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Close CSV file"""
        if self.file is not None:
            self.file.close()


class TensorBoardLogger(Callback):
    """
    Log training metrics to TensorBoard
    """
    
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.SummaryWriter = SummaryWriter
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.SummaryWriter = None
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize TensorBoard writer"""
        if self.SummaryWriter is not None:
            self.writer = self.SummaryWriter(self.log_dir)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log metrics to TensorBoard"""
        if logs is None or self.writer is None:
            return
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch + 1)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Close TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()


class ProgressBar(Callback):
    """
    Display training progress with a progress bar
    """
    
    def __init__(self, show_epoch_progress: bool = True, show_batch_progress: bool = False):
        super().__init__()
        self.show_epoch_progress = show_epoch_progress
        self.show_batch_progress = show_batch_progress
        self.epoch_start_time = None
        self.total_epochs = None
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize progress tracking"""
        if self.trainer is not None:
            self.total_epochs = self.trainer.epochs
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Start epoch timing"""
        if self.show_epoch_progress:
            self.epoch_start_time = time.time()
            print(f"Epoch {epoch + 1}/{self.total_epochs}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Display epoch results"""
        if self.show_epoch_progress and self.epoch_start_time is not None:
            elapsed = time.time() - self.epoch_start_time
            
            # Format metrics
            metrics_str = ""
            if logs:
                metrics = [f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))]
                metrics_str = " - ".join(metrics)
            
            print(f"Epoch {epoch + 1}/{self.total_epochs} - {elapsed:.2f}s - {metrics_str}")


class LearningRateScheduler(Callback):
    """
    Custom learning rate scheduler callback
    """
    
    def __init__(self, schedule: Callable[[int], float], verbose: bool = True):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Update learning rate based on schedule"""
        if self.trainer is None:
            return
        
        # Get new learning rate
        new_lr = self.schedule(epoch)
        
        # Update optimizer
        old_lr = self.trainer.optimizer.param_groups[0]['lr']
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        if self.verbose and new_lr != old_lr:
            print(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")


class MetricsHistory(Callback):
    """
    Keep track of training metrics history
    """
    
    def __init__(self):
        super().__init__()
        self.history = {}
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Record metrics for this epoch"""
        if logs is None:
            return
        
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def plot_metrics(self, metrics: Optional[List[str]] = None, figsize: Tuple[int, int] = (12, 8)):
        """Plot training metrics"""
        if not self.history:
            print("No metrics to plot")
            return
        
        if metrics is None:
            metrics = list(self.history.keys())
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in self.history]
        
        if not available_metrics:
            print("No available metrics to plot")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        cols = min(2, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is not None:
                epochs = range(1, len(self.history[metric]) + 1)
                ax.plot(epochs, self.history[metric])
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.grid(True)
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def save_history(self, filepath: str):
        """Save metrics history to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self.history, f)
    
    def load_history(self, filepath: str):
        """Load metrics history from file"""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                self.history = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                self.history = pickle.load(f)


# Callback manager
class CallbackManager:
    """
    Manages multiple callbacks during training
    """
    
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
        self.trainer = None
    
    def set_trainer(self, trainer):
        """Set trainer for all callbacks"""
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


# Pre-configured callback collections
def get_default_callbacks(checkpoint_dir: str = "models/checkpoints", 
                         log_dir: str = "logs") -> List[Callback]:
    """Get a default set of useful callbacks"""
    return [
        EarlyStopping(monitor='val_loss', patience=10, verbose=True),
        ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=True),
        ModelCheckpoint(
            filepath=f"{checkpoint_dir}/checkpoint_epoch_{{epoch:02d}}_{{val_loss:.4f}}.pth",
            monitor='val_loss',
            save_best_only=True,
            verbose=True
        ),
        CSVLogger(f"{log_dir}/training_log.csv"),
        ProgressBar(show_epoch_progress=True),
        MetricsHistory()
    ]


def get_minimal_callbacks() -> List[Callback]:
    """Get a minimal set of callbacks for basic training"""
    return [
        ProgressBar(show_epoch_progress=True),
        MetricsHistory()
    ]