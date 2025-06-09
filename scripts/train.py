#!/usr/bin/env python3
"""
Train brain tumor segmentation model.
"""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
import numpy as np
from datetime import datetime
import torch.nn.functional as F

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
                      default=1,
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
            
            # Reduce memory usage by taking central slices only
            center_slice = mod.shape[-1] // 2
            mod = mod[..., center_slice-5:center_slice+5]  # Reduced to 10 slices
            seg = seg[..., center_slice-5:center_slice+5]
            
            # Add channel dimension to segmentation
            seg = np.expand_dims(seg, axis=0)
            
            # Debug shapes
            print(f"Loaded case {case_id}:")
            print(f"  Modalities shape: {mod.shape}")
            print(f"  Segmentation shape: {seg.shape}")
            
            modalities.append(mod)
            segmentations.append(seg)
    
    X = np.stack(modalities)
    y = np.stack(segmentations)
    
    print("\nFinal dataset shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape: {data.shape}")
        print(f"  Target shape: {target.shape}")
        
        data = data.to(device).float()  # Ensure float32
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        if isinstance(output, tuple):
            # Handle deep supervision
            main_output, deep3_output, deep2_output = output
            
            # Convert target to one-hot
            target_onehot = F.one_hot(target.squeeze(1).long(), num_classes=4).permute(0, 4, 1, 2, 3).float()
            print("Loss calculation shapes:")
            print(f"  pred shape: {main_output.shape}")
            print(f"  target shape: {target.shape}")
            print(f"  target_onehot shape: {target_onehot.shape}")
            
            # Calculate loss for each output
            main_loss = criterion(main_output, target_onehot)
            deep3_loss = criterion(deep3_output, target_onehot)
            deep2_loss = criterion(deep2_output, target_onehot)
            
            # Combine losses with weights
            loss = main_loss + 0.5 * deep3_loss + 0.3 * deep2_loss
            print(f"  Main loss: {main_loss:.4f}")
            print(f"  Deep3 loss: {deep3_loss:.4f}")
            print(f"  Deep2 loss: {deep2_loss:.4f}")
            print(f"  Combined loss: {loss:.4f}")
        else:
            # Convert target to one-hot
            target_onehot = F.one_hot(target.squeeze(1).long(), num_classes=4).permute(0, 4, 1, 2, 3).float()
            print("Loss calculation shapes:")
            print(f"  pred shape: {output.shape}")
            print(f"  target shape: {target.shape}")
            print(f"  target_onehot shape: {target_onehot.shape}")
            
            loss = criterion(output, target_onehot)
            print(f"  Dice loss: {loss:.4f}")
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device).float()  # Ensure float32
            target = target.to(device)
            
            # Forward pass
            output = model(data)
            
            if isinstance(output, tuple):
                # Handle deep supervision (use only main output for validation)
                output = output[0]
            
            # Convert target to one-hot
            target_onehot = F.one_hot(target.squeeze(1).long(), num_classes=4).permute(0, 4, 1, 2, 3).float()
            
            # Calculate loss
            loss = criterion(output, target_onehot)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def analyze_class_distribution(dataloader):
    """Analyze class distribution in the dataset."""
    class_counts = torch.zeros(4)
    total_voxels = 0
    
    print("\nAnalyzing class distribution...")
    for _, target in dataloader:
        for c in range(4):
            class_counts[c] += (target == c).sum().item()
        total_voxels += target.numel()
    
    print("\nClass distribution:")
    for c in range(4):
        percentage = (class_counts[c] / total_voxels) * 100
        print(f"Class {c}: {percentage:.2f}%")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    X, y = load_dataset(args.data_dir)
    
    # Convert to torch tensors
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    
    # Split data
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print("\nTraining set shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset,
                           batch_size=args.batch_size,
                           shuffle=True)
    val_loader = DataLoader(val_dataset,
                         batch_size=args.batch_size)
    
    # Analyze class distribution
    analyze_class_distribution(train_loader)
    
    # Create model
    model = get_model(device=device)
    
    # Loss and optimizer
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"\nValidation loss: {val_loss:.4f}")
        
        # Save model
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_model(model, optimizer, epoch, args.output_dir, is_best)

if __name__ == '__main__':
    main() 