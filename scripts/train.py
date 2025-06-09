#!/usr/bin/env python3
"""
Train brain tumor segmentation model.
"""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import get_model, save_model
from src.metrics import DiceLoss, evaluate_prediction

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--data-dir',
                      type=str,
                      required=True,
                      help='Path to preprocessed data directory')
    parser.add_argument('--output-dir',
                      type=str,
                      required=True,
                      help='Output directory for model checkpoints')
    parser.add_argument('--batch-size',
                      type=int,
                      default=4,
                      help='Batch size')
    parser.add_argument('--epochs',
                      type=int,
                      default=100,
                      help='Number of epochs')
    parser.add_argument('--lr',
                      type=float,
                      default=1e-4,
                      help='Learning rate')
    parser.add_argument('--device',
                      type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    return parser.parse_args()

def load_dataset(data_dir):
    """Load preprocessed data."""
    modalities = []
    segmentations = []
    
    for file in os.listdir(data_dir):
        if file.endswith('_modalities.npy'):
            case_id = file.replace('_modalities.npy', '')
            mod = np.load(os.path.join(data_dir, file))
            seg = np.load(os.path.join(data_dir, f'{case_id}_segmentation.npy'))
            modalities.append(mod)
            segmentations.append(seg)
    
    return np.stack(modalities), np.stack(segmentations)

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print('Loading data...')
    X, y = load_dataset(args.data_dir)
    
    # Split data
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Create data loaders
    train_loader = DataLoader(list(zip(X_train, y_train)),
                           batch_size=args.batch_size,
                           shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)),
                         batch_size=args.batch_size)
    
    # Initialize model and optimizer
    model = get_model(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceLoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data = data.to(args.device).float()
            target = target.to(args.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(args.device).float()
                target = target.to(args.device)
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model,
                epoch,
                optimizer,
                val_loss,
                output_dir / f'model_best.pth'
            )
        
        # Save latest model
        save_model(
            model,
            epoch,
            optimizer,
            val_loss,
            output_dir / f'model_latest.pth'
        )

if __name__ == '__main__':
    main() 