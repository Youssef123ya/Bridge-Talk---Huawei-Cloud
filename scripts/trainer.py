import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging
import time
import os
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt

from ..models.base_model import BaseModel
from ..utils.helpers import AverageMeter, format_time
from .callbacks import CallbackManager, EarlyStopping, ModelCheckpoint, LearningRateScheduler

class AdvancedTrainer:
    """Advanced trainer with callbacks, mixed precision, and comprehensive logging"""

    def __init__(self,
                 model: BaseModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 device: str = 'cuda',
                 save_dir: str = 'models/checkpoints',
                 experiment_name: str = None,
                 use_mixed_precision: bool = True,
                 gradient_clip_val: float = 1.0,
                 accumulation_steps: int = 1):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.gradient_clip_val = gradient_clip_val
        self.accumulation_steps = accumulation_steps

        # Mixed precision setup
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed precision training enabled")

        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        # Callback manager
        self.callback_manager = CallbackManager()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        self.training_start_time = None

        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # Metrics tracking
        self.epoch_metrics = defaultdict(list)

        print(f"🚀 Trainer initialized for experiment: {self.experiment_name}")
        print(f"   Model: {model.__class__.__name__}")
        print(f"   Parameters: {model.get_num_parameters():,}")
        print(f"   Device: {device}")
        print(f"   Mixed precision: {self.use_mixed_precision}")

    def setup_logging(self):
        """Setup comprehensive logging"""

        log_dir = Path('logs') / self.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler
        log_file = log_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

        print(f"📝 Logging to: {log_file}")

    def add_callback(self, callback):
        """Add a callback to the trainer"""
        self.callback_manager.add_callback(callback)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with advanced features"""

        self.model.train()

        # Metrics tracking
        losses = AverageMeter('Loss')
        accuracies = AverageMeter('Accuracy')
        batch_times = AverageMeter('Batch Time')

        end = time.time()

        # Call epoch start callbacks
        self.callback_manager.on_epoch_start(self.current_epoch, {})

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Call batch start callbacks
            batch_logs = {'batch': batch_idx, 'size': len(data)}
            self.callback_manager.on_batch_start(batch_idx, batch_logs)

            # Forward pass with optional mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target) / self.accumulation_steps
            else:
                output = self.model(data)
                loss = self.criterion(output, target) / self.accumulation_steps

            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(self.train_loader) - 1:

                # Gradient clipping
                if self.gradient_clip_val > 0:
                    if self.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip_val
                    )

                # Optimizer step
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(target)

            # Update metrics
            losses.update(loss.item() * self.accumulation_steps, len(target))
            accuracies.update(accuracy, len(target))
            batch_times.update(time.time() - end)

            # Batch end callbacks
            batch_logs.update({
                'loss': loss.item() * self.accumulation_steps,
                'accuracy': accuracy,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            self.callback_manager.on_batch_end(batch_idx, batch_logs)

            end = time.time()

            # Logging
            if batch_idx % 50 == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] '
                    f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                    f'Acc: {accuracies.val:.2f}% ({accuracies.avg:.2f}%) '
                    f'Time: {batch_times.val:.3f}s'
                )

        # Call epoch end callbacks
        epoch_logs = {
            'loss': losses.avg,
            'accuracy': accuracies.avg,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        self.callback_manager.on_epoch_end(self.current_epoch, epoch_logs)

        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg,
            'batch_time': batch_times.avg
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""

        self.model.eval()

        losses = AverageMeter('Val Loss')
        accuracies = AverageMeter('Val Accuracy')

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = 100. * correct / len(target)

                # Update metrics
                losses.update(loss.item(), len(target))
                accuracies.update(accuracy, len(target))

                # Store for detailed analysis
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())

        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix

        try:
            # Class-wise metrics
            report = classification_report(
                all_targets, all_predictions, 
                output_dict=True, zero_division=0
            )

            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']

        except Exception as e:
            self.logger.warning(f"Could not calculate detailed metrics: {e}")
            weighted_precision = weighted_recall = weighted_f1 = 0.0

        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg,
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1,
            'predictions': all_predictions,
            'targets': all_targets
        }

    def train(self,
              epochs: int,
              save_best_only: bool = True,
              monitor_metric: str = 'val_accuracy',
              mode: str = 'max') -> Dict[str, List]:
        """Complete training loop with callbacks"""

        self.training_start_time = time.time()

        # Training start callbacks
        self.callback_manager.on_train_start({})

        self.logger.info(f"🚀 Starting training for {epochs} epochs")
        self.logger.info(f"   Monitor metric: {monitor_metric} ({mode})")
        self.logger.info(f"   Save best only: {save_best_only}")

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                self.logger.info(f"\n📅 Epoch {epoch + 1}/{epochs}")
                print(f"\n📅 Epoch {epoch + 1}/{epochs}")

                # Training phase
                train_metrics = self.train_epoch()

                # Validation phase
                val_metrics = self.validate_epoch()

                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if monitor_metric.startswith('val_'):
                            metric_value = val_metrics.get(monitor_metric.replace('val_', ''), 0)
                        else:
                            metric_value = train_metrics.get(monitor_metric.replace('train_', ''), 0)
                        self.scheduler.step(metric_value)
                    else:
                        self.scheduler.step()

                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time

                # Update history
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['learning_rate'].append(current_lr)
                self.history['epoch_time'].append(epoch_time)

                # Log metrics
                self.logger.info(
                    f"   Train - Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.2f}%"
                )
                self.logger.info(
                    f"   Val   - Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['accuracy']:.2f}%, "
                    f"F1: {val_metrics['f1_score']:.4f}"
                )
                self.logger.info(f"   Time: {format_time(epoch_time)}, LR: {current_lr:.6f}")

                print(f"   Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
                print(f"   Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
                print(f"   Time: {format_time(epoch_time)}, LR: {current_lr:.6f}")

                # Determine if this is the best model
                if monitor_metric.startswith('val_'):
                    current_score = val_metrics.get(monitor_metric.replace('val_', ''), 0)
                else:
                    current_score = train_metrics.get(monitor_metric.replace('train_', ''), 0)

                is_best = False
                if mode == 'max' and current_score > self.best_val_score:
                    self.best_val_score = current_score
                    is_best = True
                elif mode == 'min' and (self.best_val_score == 0 or current_score < self.best_val_score):
                    self.best_val_score = current_score
                    is_best = True

                # Save checkpoint
                if is_best or not save_best_only:
                    self.save_checkpoint(
                        epoch + 1,
                        train_metrics,
                        val_metrics,
                        is_best=is_best
                    )

                # Store epoch metrics for callbacks
                epoch_logs = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1_score': val_metrics['f1_score'],
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time,
                    'is_best': is_best
                }

                # Check if training should stop (callbacks handle this)
                if self.callback_manager.should_stop_training():
                    self.logger.info("🛑 Training stopped by callback")
                    print("🛑 Training stopped by callback")
                    break

        except KeyboardInterrupt:
            self.logger.info("⏹️  Training interrupted by user")
            print("⏹️  Training interrupted by user")

        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            print(f"❌ Training failed: {e}")
            raise

        finally:
            # Training end callbacks
            self.callback_manager.on_train_end({})

            total_time = time.time() - self.training_start_time
            self.logger.info(f"\n✅ Training completed in {format_time(total_time)}")
            self.logger.info(f"   Best {monitor_metric}: {self.best_val_score:.4f}")

            print(f"\n✅ Training completed in {format_time(total_time)}")
            print(f"   Best {monitor_metric}: {self.best_val_score:.4f}")

        return self.history

    def save_checkpoint(self,
                       epoch: int,
                       train_metrics: Dict[str, float],
                       val_metrics: Dict[str, float],
                       is_best: bool = False):
        """Save model checkpoint with comprehensive information"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'history': self.history,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_config': self.model.model_config,
            'experiment_name': self.experiment_name
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save regular checkpoint
        filename = f'checkpoint_epoch_{epoch}.pth'
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        # Save best model
        if is_best:
            best_filepath = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_filepath)
            self.logger.info(f"💾 New best model saved: {best_filepath}")
            print(f"💾 New best model saved with {val_metrics.get('accuracy', 0):.2f}% accuracy")

        # Save latest model
        latest_filepath = self.save_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_filepath)

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True, load_scheduler: bool = True):
        """Load model checkpoint"""

        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if load_scheduler and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler state
        if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_score = checkpoint.get('best_val_score', 0.0)
        self.history = checkpoint.get('history', self.history)

        self.logger.info(f"📂 Checkpoint loaded from {filepath}")
        self.logger.info(f"   Resuming from epoch {self.current_epoch + 1}")
        print(f"📂 Checkpoint loaded: epoch {checkpoint.get('epoch', 0)}")

    def create_training_plots(self, save_dir: str = None):
        """Create comprehensive training plots"""

        if save_dir is None:
            save_dir = f'logs/{self.experiment_name}/plots'

        os.makedirs(save_dir, exist_ok=True)

        # Create training history plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {self.experiment_name}', fontsize=16)

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate plot
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # Epoch time plot
        axes[1, 1].plot(epochs, self.history['epoch_time'], 'm-')
        axes[1, 1].set_title('Epoch Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Training plots saved to {plot_path}")

        return plot_path

def create_trainer_from_config(model: BaseModel,
                             train_loader: DataLoader,
                             val_loader: DataLoader,
                             config: Dict[str, Any]) -> AdvancedTrainer:
    """Factory function to create trainer from configuration"""

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer_name = config.get('optimizer', 'adam').lower()
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Scheduler
    scheduler = None
    scheduler_config = config.get('scheduler', {})
    if scheduler_config:
        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')

        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                verbose=True
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get('epochs', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )

    # Device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config.get('save_dir', 'models/checkpoints'),
        experiment_name=config.get('experiment_name'),
        use_mixed_precision=config.get('use_mixed_precision', True),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        accumulation_steps=config.get('accumulation_steps', 1)
    )

    # Add default callbacks
    if config.get('early_stopping', True):
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=config.get('early_stopping_patience', 10),
            mode='max',
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )
        trainer.add_callback(early_stopping)

    if config.get('model_checkpoint', True):
        model_checkpoint = ModelCheckpoint(
            filepath=trainer.save_dir / 'best_model.pth',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
        trainer.add_callback(model_checkpoint)

    return trainer
