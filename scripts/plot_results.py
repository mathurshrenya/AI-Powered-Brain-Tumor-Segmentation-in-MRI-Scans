#!/usr/bin/env python3
"""
Plot and save visualization results.
"""
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import get_model, load_model
from src.visualization import plot_segmentation, create_montage
from src.data import load_data

def parse_args():
    parser = argparse.ArgumentParser(description='Plot segmentation results')
    parser.add_argument('--data-dir',
                      type=str,
                      required=True,
                      help='Path to preprocessed data directory')
    parser.add_argument('--model-path',
                      type=str,
                      required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--output-dir',
                      type=str,
                      required=True,
                      help='Directory to save plots')
    parser.add_argument('--num-samples',
                      type=int,
                      default=5,
                      help='Number of samples to plot')
    return parser.parse_args()

def main():
    """Main plotting function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    _, test_loader = load_data(args.data_dir, batch_size=1, train_val_split=0.9)
    
    # Create and load model
    model = get_model(device=device)
    load_model(model, args.model_path)
    model.eval()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= args.num_samples:
                break
                
            print(f"\nProcessing sample {i+1}/{args.num_samples}")
            
            # Move data to device
            data = data.to(device)
            target = target.to(device)
            
            # Get model prediction
            output = model(data)
            
            # Convert to numpy arrays
            data = data.cpu().numpy()
            target = target.cpu().numpy()
            pred = output.cpu().numpy()
            
            # Get middle slices for visualization
            slice_idx = data.shape[-1] // 2
            
            # Plot each modality
            for modality in range(data.shape[1]):
                # Get slices
                image = data[0, modality, :, :, slice_idx]
                true_mask = target[0, 0, :, :, slice_idx]
                pred_mask = np.argmax(pred[0], axis=0)[:, :, slice_idx]
                
                # Create plot
                fig = plot_segmentation(image, true_mask, pred_mask)
                
                # Save plot
                save_path = output_dir / f'sample_{i+1}_modality_{modality}.png'
                fig.savefig(save_path)
                plt.close(fig)
            
            print(f"Saved plots for sample {i+1}")
    
    print(f"\nPlots saved to: {args.output_dir}")

if __name__ == '__main__':
    main() 