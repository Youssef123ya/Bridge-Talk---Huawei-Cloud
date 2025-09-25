import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import logging
import os
import time
from pathlib import Path

class Callback:
    """Base callback class"""

    def on_train_start(self, logs: Dict[str, Any] = None):
        """Called at the beginning of training"""
        pass

    def on_train_end(self, logs: Dict[str, Any] = None):
        """Called at the end of training"""
        pass

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of an epoch"""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the end of an epoch"""
        pass

    def on_batch_start(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of a batch"""
        pass

    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the end of a batch"""
        pass

class CallbackManager:
    """Manages multiple callbacks"""

    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []
        self._stop_training = False

    def add_callback(self, callback: Callback):
        """Add a callback"""
        self.callbacks.append(callback)

    def should_stop_training(self) -> bool:
        """Check if training should stop"""
        return self._stop_training

    def stop_training(self):
        """Signal to stop training"""
        self._stop_training = True

    def on_train_start(self, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_train_start(logs)

    def on_train_end(self, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_start(self, batch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_batch_start(batch, logs)

    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

class EarlyStopping(Callback):
    """Early stopping callback to prevent overfitting"""

    def __init__(self, 
                 monitor: str = 'val_loss',
                 patience: int = 10,
                 mode: str = 'min',
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True,
                 verbose: bool = True):

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_score = None
        self.patience_counter = 0
        self.best_weights = None
        self.stopped_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode {mode} is unknown, use 'min' or 'max'")

    def on_train_start(self, logs: Dict[str, Any] = None):
        self.patience_counter = 0
        self.best_score = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        current_score = logs.get(self.monitor)

        if current_score is None:
            if self.verbose:
                print(f"⚠️  Early stopping metric '{self.monitor}' not found")
            return

        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                # Store current weights (this would need access to model)
                pass
        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.patience_counter = 0
            if self.restore_best_weights:
                # Store current weights
                pass
        else:
            self.patience_counter += 1

            if self.patience_counter >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered after {self.patience} epochs without improvement")
                    print(f"   Best {self.monitor}: {self.best_score:.4f}")

                # Signal to stop training
                # This requires access to the callback manager
                # In practice, this would be handled by the trainer
                return True  # Signal to stop

        if self.verbose and self.patience_counter > 0:
            print(f"   Early stopping: {self.patience_counter}/{self.patience}")

    def on_train_end(self, logs: Dict[str, Any] = None):
        if self.stopped_epoch > 0 and self.verbose:
            print(f"\n📊 Training stopped at epoch {self.stopped_epoch + 1}")

class ModelCheckpoint(Callback):
    """Save model checkpoints during training"""

    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_weights_only: bool = False,
                 verbose: bool = True):

        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self.best_score = None

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode {mode} is unknown, use 'min' or 'max'")

        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        current_score = logs.get(self.monitor)

        if current_score is None:
            if self.verbose:
                print(f"⚠️  Checkpoint metric '{self.monitor}' not found")
            return

        if self.best_score is None:
            self.best_score = current_score
            should_save = True
        elif self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            should_save = True
        else:
            should_save = not self.save_best_only

        if should_save:
            if self.verbose:
                print(f"\n💾 Saving checkpoint to {self.filepath}")

            # In practice, this would save the model
            # The trainer handles the actual saving

class LearningRateScheduler(Callback):
    """Custom learning rate scheduling callback"""

    def __init__(self, 
                 schedule: Callable[[int], float],
                 verbose: bool = True):

        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        new_lr = self.schedule(epoch)

        # In practice, this would update the optimizer learning rate
        # The trainer handles the actual LR updates

        if self.verbose:
            print(f"📈 Learning rate for epoch {epoch + 1}: {new_lr:.6f}")

class ReduceLROnPlateau(Callback):
    """Reduce learning rate when metric has stopped improving"""

    def __init__(self,
                 monitor: str = 'val_loss',
                 factor: float = 0.5,
                 patience: int = 5,
                 mode: str = 'min',
                 min_delta: float = 1e-4,
                 min_lr: float = 1e-7,
                 verbose: bool = True):

        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.verbose = verbose

        self.best_score = None
        self.patience_counter = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode {mode} is unknown, use 'min' or 'max'")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        current_score = logs.get(self.monitor)

        if current_score is None:
            return

        if self.best_score is None:
            self.best_score = current_score
        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

            if self.patience_counter >= self.patience:
                # In practice, this would reduce the learning rate
                # The trainer handles the actual LR updates

                if self.verbose:
                    print(f"\n📉 Reducing learning rate by factor {self.factor}")

                self.patience_counter = 0

class CSVLogger(Callback):
    """Log training metrics to CSV file"""

    def __init__(self, 
                 filename: str,
                 separator: str = ',',
                 append: bool = False):

        self.filename = Path(filename)
        self.separator = separator
        self.append = append
        self.keys = None

        # Create directory if it doesn't exist
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        # Create or clear file
        if not append or not self.filename.exists():
            with open(self.filename, 'w') as f:
                pass  # Create empty file

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        def handle_value(k):
            is_zero_d_tensor = isinstance(logs[k], torch.Tensor) and logs[k].ndim == 0
            if isinstance(logs[k], (int, float)) or is_zero_d_tensor:
                return logs[k]
            return str(logs[k])

        if self.keys is None:
            self.keys = ['epoch'] + sorted(logs.keys())

            # Write header
            with open(self.filename, 'a') as f:
                f.write(self.separator.join(self.keys) + '\n')

        # Write data
        row_data = [str(epoch)]
        for key in self.keys[1:]:  # Skip 'epoch'
            if key in logs:
                row_data.append(str(handle_value(key)))
            else:
                row_data.append('')

        with open(self.filename, 'a') as f:
            f.write(self.separator.join(row_data) + '\n')

class TensorBoardLogger(Callback):
    """Log training metrics to TensorBoard"""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            print("⚠️  TensorBoard not available. Install with: pip install tensorboard")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if self.writer is None:
            return

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

    def on_train_end(self, logs: Dict[str, Any] = None):
        if self.writer:
            self.writer.close()

class WandbLogger(Callback):
    """Log training metrics to Weights & Biases"""

    def __init__(self, 
                 project: str,
                 name: str = None,
                 config: Dict[str, Any] = None):

        self.project = project
        self.name = name
        self.config = config
        self.wandb = None

        try:
            import wandb
            self.wandb = wandb
            wandb.init(project=project, name=name, config=config)
        except ImportError:
            print("⚠️  Wandb not available. Install with: pip install wandb")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if self.wandb is None:
            return

        wandb_logs = {'epoch': epoch}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                wandb_logs[key] = value

        self.wandb.log(wandb_logs)

    def on_train_end(self, logs: Dict[str, Any] = None):
        if self.wandb:
            self.wandb.finish()

class ProgressCallback(Callback):
    """Display training progress with detailed information"""

    def __init__(self, update_freq: int = 1):
        self.update_freq = update_freq
        self.epoch_start_time = None
        self.train_start_time = None

    def on_train_start(self, logs: Dict[str, Any] = None):
        self.train_start_time = time.time()
        print("🚀 Training started!")

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        self.epoch_start_time = time.time()
        print(f"\n📅 Starting epoch {epoch + 1}")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time

            # Format metrics for display
            metrics_str = []
            for key, value in logs.items():
                if isinstance(value, float):
                    metrics_str.append(f"{key}: {value:.4f}")
                elif isinstance(value, int):
                    metrics_str.append(f"{key}: {value}")

            print(f"⏱️  Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            if metrics_str:
                print(f"   📊 {' | '.join(metrics_str)}")

    def on_train_end(self, logs: Dict[str, Any] = None):
        if self.train_start_time:
            total_time = time.time() - self.train_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)

            print(f"\n✅ Training completed!")
            print(f"   ⏱️  Total time: {hours}h {minutes}m {seconds}s")

class GradientClipping(Callback):
    """Gradient clipping callback"""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        # In practice, gradient clipping is handled in the trainer
        # This is just for demonstration
        pass

class LRFinder(Callback):
    """Learning rate finder callback"""

    def __init__(self, 
                 start_lr: float = 1e-7,
                 end_lr: float = 1e-1,
                 num_epochs: int = 5):

        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs
        self.lr_schedule = []
        self.losses = []
        self.current_lr = start_lr

    def on_train_start(self, logs: Dict[str, Any] = None):
        # Generate LR schedule
        lr_mult = (self.end_lr / self.start_lr) ** (1.0 / self.num_epochs)

        for epoch in range(self.num_epochs):
            self.lr_schedule.append(self.start_lr * (lr_mult ** epoch))

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        if epoch < len(self.lr_schedule):
            self.current_lr = self.lr_schedule[epoch]
            # In practice, this would update the optimizer LR

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if 'loss' in logs:
            self.losses.append(logs['loss'])

    def plot_lr_finder(self, save_path: str = None):
        """Plot learning rate vs loss"""
        import matplotlib.pyplot as plt

        if len(self.losses) == 0:
            print("⚠️  No loss data to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.lr_schedule[:len(self.losses)], self.losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 LR finder plot saved to {save_path}")

        plt.show()

def create_default_callbacks(config: Dict[str, Any]) -> List[Callback]:
    """Create a set of default callbacks based on configuration"""

    callbacks = []

    # Progress callback
    callbacks.append(ProgressCallback())

    # Early stopping
    if config.get('early_stopping', True):
        early_stopping = EarlyStopping(
            monitor=config.get('early_stopping_monitor', 'val_loss'),
            patience=config.get('early_stopping_patience', 10),
            mode=config.get('early_stopping_mode', 'min'),
            verbose=True
        )
        callbacks.append(early_stopping)

    # CSV logging
    if config.get('csv_logging', True):
        log_dir = config.get('log_dir', 'logs')
        csv_logger = CSVLogger(
            filename=os.path.join(log_dir, 'training_log.csv')
        )
        callbacks.append(csv_logger)

    # TensorBoard logging
    if config.get('tensorboard_logging', False):
        log_dir = config.get('log_dir', 'logs')
        tb_logger = TensorBoardLogger(
            log_dir=os.path.join(log_dir, 'tensorboard')
        )
        callbacks.append(tb_logger)

    # Wandb logging
    if config.get('wandb_logging', False):
        wandb_logger = WandbLogger(
            project=config.get('wandb_project', 'arasl-recognition'),
            name=config.get('experiment_name'),
            config=config
        )
        callbacks.append(wandb_logger)

    return callbacks
