#!/usr/bin/env python3
"""
Evaluate trained model on test data.
"""
import os
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import get_model, load_model
from src.metrics import evaluate_prediction
from src.visualization import plot_segmentation

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
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
                      help='Output directory for evaluation results')
    parser.add_argument('--device',
                      type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    return parser.parse_args()

def load_test_data(data_dir):
    """Load test data."""
    modalities = []
    segmentations = []
    
    for file in os.listdir(data_dir):
        if file.endswith('_modalities.npy'):
            case_id = file.replace('_modalities.npy', '')
            mod = np.load(os.path.join(data_dir, file))
            seg = np.load(os.path.join(data_dir, f'{case_id}_segmentation.npy'))
            
            # Add channel dimension to segmentation
            seg = np.expand_dims(seg, axis=0)
            
            modalities.append(mod)
            segmentations.append(seg)
    
    # Use last case as test data
    X_test = modalities[-1:]
    y_test = segmentations[-1:]
    
    return np.stack(X_test), np.stack(y_test)

def save_results(metrics, output_dir):
    """Save evaluation metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f'{metric_name}: {value:.4f}\n')

def main():
    args = parse_args()
    
    # Load model
    model = get_model(device=args.device)
    model, _ = load_model(args.model_path, model)
    model.eval()
    
    # Load test data
    print('Loading test data...')
    X_test, y_test = load_test_data(args.data_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('Evaluating...')
    all_metrics = []
    
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(zip(X_test, y_test))):
            # Convert to tensor
            X = torch.from_numpy(X).unsqueeze(0).float().to(args.device)
            
            # Get prediction
            pred = model(X)
            pred = torch.softmax(pred, dim=1)
            pred = pred.cpu().numpy()
            
            # Calculate metrics
            metrics = evaluate_prediction(pred[0], y)  # Remove batch dimension from pred only
            all_metrics.append(metrics)
            
            # Save example visualization
            mid_slice = y.shape[-1] // 2
            fig = plot_segmentation(
                image=X[0, 0, :, :, mid_slice].cpu().numpy(),
                true_mask=y[0, :, :, mid_slice],
                pred_mask=np.argmax(pred[0, :, :, :, mid_slice], axis=0)
            )
            plt.savefig(output_dir / f'example_{i}.png')
            plt.close()
    
    # Calculate mean metrics
    mean_metrics = {}
    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics]
        mean_metrics[metric] = np.mean(values)
    
    # Save results
    save_results(mean_metrics, output_dir)
    print('\nResults saved to:', output_dir)
    
    # Print summary
    print('\nEvaluation metrics:')
    for metric_name, value in mean_metrics.items():
        print(f'{metric_name}: {value:.4f}')

if __name__ == '__main__':
    main() 