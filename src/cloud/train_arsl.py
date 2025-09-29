"""
Cloud-ready training script for Arabic Sign Language Recognition
Compatible with Huawei Cloud ModelArts
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Configure logging for cloud environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class ARSLDataset(Dataset):
    """
    Arabic Sign Language Dataset for cloud training
    """
    
    def __init__(self, data_path: str, labels_file: str, transform=None, split: str = 'train'):
        """
        Initialize dataset
        
        Args:
            data_path: Path to data directory (OBS mount or local)
            labels_file: Path to labels CSV file
            transform: Image transformations
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # Load labels
        labels_df = pd.read_csv(labels_file)
        
        # Filter by split if column exists
        if 'split' in labels_df.columns:
            labels_df = labels_df[labels_df['split'] == split]
        
        self.labels_df = labels_df.reset_index(drop=True)
        
        # Create label mapping
        unique_labels = sorted(labels_df['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        
        logger.info(f"Loaded {len(self.labels_df)} samples for {split} split")
        logger.info(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        """Get sample by index"""
        row = self.labels_df.iloc[idx]
        
        # Load image
        img_path = self.data_path / row['filename']
        if not img_path.exists():
            # Try alternative path structure
            img_path = self.data_path / row['label'] / row['filename']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (64, 64), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.label_to_idx[row['label']]
        
        return image, label


class SimpleCNN(nn.Module):
    """
    CNN model for Arabic Sign Language Recognition
    Optimized for cloud training
    """
    
    def __init__(self, num_classes: int = 32, dropout: float = 0.5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CloudTrainer:
    """
    Cloud-optimized trainer for ARSL model
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SimpleCNN(
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def get_data_transforms(self):
        """Get data transformations for training and validation"""
        train_transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_data_loaders(self, data_path: str, labels_file: str) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        train_transform, val_transform = self.get_data_transforms()
        
        # Create datasets
        train_dataset = ARSLDataset(
            data_path=data_path,
            labels_file=labels_file,
            transform=train_transform,
            split='train'
        )
        
        val_dataset = ARSLDataset(
            data_path=data_path,
            labels_file=labels_file,
            transform=val_transform,
            split='val'
        )
        
        # If no split column, create manual split
        if len(val_dataset) == 0:
            logger.info("No validation split found, creating manual split")
            full_dataset = ARSLDataset(
                data_path=data_path,
                labels_file=labels_file,
                transform=train_transform
            )
            
            # 80-20 split
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
            # Update transform for validation set
            val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}\"")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log batch metrics for cloud monitoring
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, output_path: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = os.path.join(output_path, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model: {best_model_path}")
    
    def train(self, data_path: str, labels_file: str, output_path: str):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Configuration: {self.config}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(data_path, labels_file)
        
        # Training loop
        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_model(output_path, epoch, is_best)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}, "
                f"Best Val Acc: {self.best_val_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping
            if len(self.val_accuracies) > self.config.get('patience', 10):
                recent_accs = self.val_accuracies[-self.config.get('patience', 10):]
                if max(recent_accs) <= self.best_val_acc * 0.99:  # No significant improvement
                    logger.info("Early stopping triggered")
                    break
        
        # Save final model and metrics
        self.save_final_results(output_path)
        logger.info("Training completed!")
    
    def save_final_results(self, output_path: str):
        """Save final training results and metrics"""
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        history_path = os.path.join(output_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save model for inference
        if self.best_model_state:
            inference_model = {
                'model_state_dict': self.best_model_state,
                'config': self.config,
                'accuracy': self.best_val_acc
            }
            
            inference_path = os.path.join(output_path, 'model_for_inference.pth')
            torch.save(inference_model, inference_path)
            
        logger.info(f"Saved final results to {output_path}")


def parse_args():
    """Parse command line arguments for cloud training"""
    parser = argparse.ArgumentParser(description='Arabic Sign Language Recognition - Cloud Training')
    
    # Data arguments
    parser.add_argument('--data_url', type=str, required=True, help='Path to training data')
    parser.add_argument('--train_url', type=str, required=True, help='Path to save outputs')
    parser.add_argument('--labels_file', type=str, default='ArSL_Data_Labels.csv', help='Labels file name')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=32, help='Number of classes')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    return parser.parse_args()


def main():
    """Main training function for cloud deployment"""
    args = parse_args()
    
    # Create training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'patience': args.patience
    }
    
    # Setup paths
    data_path = os.path.join(args.data_url, 'raw')
    labels_file = os.path.join(args.data_url, args.labels_file)
    
    # Verify data exists
    if not os.path.exists(data_path):
        logger.error(f"Data path not found: {data_path}")
        sys.exit(1)
    
    if not os.path.exists(labels_file):
        logger.error(f"Labels file not found: {labels_file}")
        sys.exit(1)
    
    # Initialize trainer
    trainer = CloudTrainer(config)
    
    # Start training
    try:
        trainer.train(data_path, labels_file, args.train_url)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()