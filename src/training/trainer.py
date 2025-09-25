"""
Advanced Training Module for Sign Language Recognition
Provides comprehensive training functionality with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple, Callable
import time
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import logging

from ..models.base_model import BaseSignLanguageModel
from ..utils.helpers import get_device, create_directories


class AdvancedTrainer:
    """
    Advanced trainer for sign language recognition models
    """
    
    def __init__(self,
                 model: BaseSignLanguageModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 checkpoint_dir: str = 'models/checkpoints',
                 log_dir: str = 'logs/experiments'):
        """
        Initialize advanced trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            experiment_name: Name for this experiment
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup criterion
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
            
        # Setup scheduler
        self.scheduler = scheduler
        
        # Experiment tracking
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        create_directories([str(self.checkpoint_dir), str(self.log_dir)])
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progress tracking
        batch_losses = []
        batch_accuracies = []
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            # Batch metrics
            batch_loss = loss.item()
            batch_accuracy = (predicted == targets).float().mean().item() * 100
            
            batch_losses.append(batch_loss)
            batch_accuracies.append(batch_accuracy)
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                    f'Loss: {batch_loss:.4f}, Acc: {batch_accuracy:.2f}%'
                )
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = (correct_predictions / total_samples) * 100
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'batch_losses': batch_losses,
            'batch_accuracies': batch_accuracies
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
                
                # Store for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = (correct_predictions / total_samples) * 100
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def train(self, 
              epochs: int,
              save_best_only: bool = True,
              monitor_metric: str = 'val_accuracy',
              early_stopping_patience: Optional[int] = None,
              save_frequency: int = 10) -> Dict[str, List]:
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train
            save_best_only: Save only the best model
            monitor_metric: Metric to monitor for best model
            early_stopping_patience: Patience for early stopping
            save_frequency: How often to save checkpoints
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        # Early stopping setup
        early_stopping_counter = 0
        best_metric_value = 0.0 if 'accuracy' in monitor_metric else float('inf')
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(current_lr)
            
            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            
            # Logging
            self.logger.info(
                f"Epoch {self.current_epoch}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            current_metric_value = val_metrics['accuracy']
            is_best = current_metric_value > best_metric_value
            
            if is_best:
                best_metric_value = current_metric_value
                self.best_val_accuracy = current_metric_value
                early_stopping_counter = 0
                
                if save_best_only:
                    self.save_checkpoint(is_best=True)
            else:
                early_stopping_counter += 1
            
            # Regular checkpoint saving
            if not save_best_only and (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping check
            if early_stopping_patience and early_stopping_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
        
        # Save final checkpoint
        self.save_checkpoint(is_best=False, suffix='final')
        
        # Save training history
        self.save_training_history()
        
        return self.history
    
    def save_checkpoint(self, is_best: bool = False, suffix: str = '') -> str:
        """Save model checkpoint"""
        
        checkpoint_data = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'history': self.history,
            'model_info': self.model.get_model_info(),
            'experiment_name': self.experiment_name
        }
        
        # Determine filename
        if is_best:
            filename = f'{self.experiment_name}_best.pth'
        elif suffix:
            filename = f'{self.experiment_name}_{suffix}.pth'
        else:
            filename = f'{self.experiment_name}_epoch_{self.current_epoch}.pth'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.history = checkpoint.get('history', {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []
        })
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resumed from epoch {self.current_epoch}")
        
        return checkpoint
    
    def save_training_history(self) -> str:
        """Save training history to JSON"""
        history_path = self.log_dir / f'{self.experiment_name}_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"Training history saved: {history_path}")
        return str(history_path)
    
    def create_training_plots(self) -> str:
        """Create training plots"""
        if not self.history['train_loss']:
            self.logger.warning("No training history to plot")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Progress - {self.experiment_name}')
        
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
        if self.history['learning_rates']:
            axes[1, 0].plot(epochs, self.history['learning_rates'], 'g-')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Validation accuracy trend
        if len(self.history['val_acc']) > 1:
            axes[1, 1].plot(epochs, self.history['val_acc'], 'r-', marker='o', markersize=3)
            axes[1, 1].axhline(y=max(self.history['val_acc']), color='g', linestyle='--', 
                              label=f'Best: {max(self.history["val_acc"]):.2f}%')
            axes[1, 1].set_title('Validation Accuracy Trend')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Validation Accuracy (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / f'{self.experiment_name}_training_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved: {plot_path}")
        return str(plot_path)


def create_trainer_from_config(model: BaseSignLanguageModel,
                              train_loader: DataLoader,
                              val_loader: DataLoader,
                              config: Dict[str, Any]) -> AdvancedTrainer:
    """
    Create trainer from configuration dictionary
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        
    Returns:
        Configured AdvancedTrainer instance
    """
    
    # Setup optimizer
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adam').lower()
    learning_rate = optimizer_config.get('learning_rate', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Setup scheduler
    scheduler = None
    scheduler_config = config.get('scheduler', {})
    if scheduler_config.get('type'):
        scheduler_type = scheduler_config['type'].lower()
        
        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_type == 'plateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=patience, factor=factor
            )
        
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    
    # Setup loss function
    criterion_config = config.get('criterion', {})
    criterion_type = criterion_config.get('type', 'crossentropy').lower()
    
    if criterion_type == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == 'focal':
        # Placeholder for FocalLoss implementation
        criterion = nn.CrossEntropyLoss()  # Fallback
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.get('device'),
        experiment_name=config.get('experiment_name'),
        checkpoint_dir=config.get('checkpoint_dir', 'models/checkpoints'),
        log_dir=config.get('log_dir', 'logs/experiments')
    )
    
    return trainer


if __name__ == "__main__":
    print("Advanced Trainer module for Sign Language Recognition")
    print("Use this module with create_trainer_from_config() function")