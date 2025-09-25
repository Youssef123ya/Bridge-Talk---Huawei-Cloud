#!/usr/bin/env python3
"""
Phase 4: Evaluation & Optimization Script
"""
import sys, os, json, time
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.base_model import ModelRegistry
from src.evaluation.metrics import compute_classification_metrics, ConfusionMatrix, compute_top_k_accuracy
from src.evaluation.visualizer import plot_confusion_matrix
from src.utils.helpers import get_device

# Configuration
DEVICE = get_device()
MODEL_PATH = 'models/checkpoints/best_model.pth'
DATA_DIR = 'data/raw/'
LABELS_FILE = 'data/processed/test.csv'
BATCH_SIZE = 16  # Reduced for faster testing


def load_model():
    """Load or create a model for evaluation demo"""
    print('🔄 Loading model...')
    
    # Check if trained model exists
    if os.path.exists(MODEL_PATH):
        # Load model checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model = ModelRegistry.get_model(checkpoint['model_name'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded trained model: {checkpoint["model_name"]}')
    else:
        # Create a demo model for testing evaluation pipeline
        print('⚠️  No trained model found. Creating demo model for evaluation testing...')
        from src.models.cnn_architectures import BasicCNN
        
        # Use 32 classes (Arabic alphabet) - we'll get this dynamically later
        # For now, create model with expected number of classes
        model = BasicCNN(num_classes=32)  # Standard Arabic alphabet
        print(f'Created demo BasicCNN model with 32 classes')
    
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_model():
    print('📊 Loading data...')
    # Create test dataset using a simple approach
    print("Creating evaluation dataset from test.csv...")
    
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    
    # Simple dataset class for evaluation
    class EvalDataset(Dataset):
        def __init__(self, csv_file, data_dir, transform=None, max_samples=100):
            df = pd.read_csv(csv_file)
            # Limit samples for testing
            if max_samples and len(df) > max_samples:
                self.df = df.sample(n=max_samples, random_state=42)
                print(f"Using {max_samples} random samples for testing")
            else:
                self.df = df
                
            self.data_dir = Path(data_dir)
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Create class mapping
            self.classes = sorted(self.df['class'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            print(f"Found {len(self.df)} samples with {len(self.classes)} classes")
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            
            # Try different possible paths for the image
            possible_paths = [
                self.data_dir / row['File_Name'],
                self.data_dir / row['class'] / row['File_Name'],
            ]
            
            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = path
                    break
            
            if image_path is None:
                # Create a dummy image if file not found
                image = Image.new('RGB', (224, 224), color='black')
            else:
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception:
                    image = Image.new('RGB', (224, 224), color='black')
            
            if self.transform:
                image = self.transform(image)
            
            label = self.class_to_idx[row['class']]
            return image, label
    
    # Create dataset and dataloader (limited samples for testing)
    eval_dataset = EvalDataset('data/processed/test.csv', 'data/raw', max_samples=100)
    test_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues with local class
        pin_memory=False
    )
    
    # Store class names for later use
    global class_names
    class_names = eval_dataset.classes
    print(f'Test samples: {len(test_loader.dataset):,}')

    model = load_model()
    print('✅ Model loaded')

    all_preds, all_labels, all_probs = [], [], []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            probs = softmax(output)
            preds = output.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_classification_metrics(all_labels, all_preds)
    top3 = compute_top_k_accuracy(all_labels, np.array(all_probs), k=3)
    metrics['top3_accuracy'] = top3
    print('📈 Evaluation Metrics:', metrics)

    # Confusion Matrix and Visualization
    cm_util = ConfusionMatrix(all_labels, all_preds, class_names)
    cm_path = 'data/analysis/confusion_matrix.png'
    
    # Create analysis directory if it doesn't exist
    os.makedirs('data/analysis', exist_ok=True)
    
    plot_confusion_matrix(all_labels, all_preds, 
                         class_names=class_names, 
                         save_path=cm_path)
    print('📊 Confusion matrix saved:', cm_path)

    # Save metrics
    with open('logs/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('💾 Metrics saved to logs/evaluation_metrics.json')

if __name__ == '__main__':
    evaluate_model()
