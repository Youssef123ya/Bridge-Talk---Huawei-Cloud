"""
Arabic Sign Language Recognition - PyTorch Training Script
This script will be executed by ModelArts for training the CNN model
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ARSLDataset(Dataset):
    """Arabic Sign Language Dataset"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('L')  # Grayscale
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            label = self.labels[idx]
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            blank_image = torch.zeros((1, 64, 64))
            return blank_image, 0


class ARSLNet(nn.Module):
    """CNN Architecture for Arabic Sign Language Recognition"""
    
    def __init__(self, num_classes=32):
        super(ARSLNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def load_dataset(data_dir, labels_file):
    """Load dataset from OBS storage"""
    logger.info(f"Loading dataset from {data_dir}")
    
    # Read labels CSV
    labels_df = pd.read_csv(labels_file)
    logger.info(f"Loaded {len(labels_df)} samples from labels file")
    
    # Get image paths and labels
    image_paths = []
    labels = []
    
    for idx, row in labels_df.iterrows():
        img_path = os.path.join(data_dir, row['image_path'])
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(row['label'])
    
    logger.info(f"Found {len(image_paths)} valid images")
    return image_paths, labels


def get_data_transforms(image_size=64):
    """Get data augmentation transforms"""
    
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transforms, val_transforms


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0:
            logger.info(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0
    )
    
    return val_loss, val_acc, precision, recall, f1


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def main(args):
    """Main training function"""
    logger.info("=" * 60)
    logger.info("ARABIC SIGN LANGUAGE RECOGNITION - TRAINING")
    logger.info("=" * 60)
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Load dataset
    logger.info("Loading dataset...")
    data_dir = args.data_url
    labels_file = os.path.join(data_dir, 'ArSL_Data_Labels.csv')
    
    image_paths, labels = load_dataset(data_dir, labels_file)
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        test_size=args.validation_split, 
        random_state=args.random_seed,
        stratify=labels
    )
    
    logger.info(f"Training samples: {len(train_paths)}")
    logger.info(f"Validation samples: {len(val_paths)}")
    
    # Get transforms
    train_transforms, val_transforms = get_data_transforms(args.image_size)
    
    # Create datasets
    train_dataset = ARSLDataset(train_paths, train_labels, train_transforms)
    val_dataset = ARSLDataset(val_paths, val_labels, val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = ARSLNet(num_classes=args.num_classes).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    patience_counter = 0
    training_history = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, precision, recall, f1 = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'epoch_time': epoch_time
        }
        training_history.append(metrics)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        logger.info(f"Epoch Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(args.train_url, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, metrics, best_model_path)
            logger.info(f"✅ New best model! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                args.train_url, 
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(args.train_url, 'final_model.pth')
    save_checkpoint(model, optimizer, epoch, metrics, final_model_path)
    
    # Save training history
    history_path = os.path.join(args.train_url, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Total Training Time: {sum(m['epoch_time'] for m in training_history):.2f}s")
    logger.info(f"Models saved to: {args.train_url}")
    
    return best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARSL Training')
    
    # Data parameters
    parser.add_argument('--data_url', type=str, required=True, help='Path to dataset')
    parser.add_argument('--train_url', type=str, required=True, help='Path to save models')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=32, help='Number of classes')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    try:
        best_acc = main(args)
        sys.exit(0 if best_acc > 85.0 else 1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
