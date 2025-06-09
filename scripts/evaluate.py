#!/usr/bin/env python3
"""
Evaluate trained brain tumor segmentation model.
"""
import os
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import sys
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import get_model, load_model
from src.metrics import evaluate_prediction
from src.visualization import (
    plot_prediction_comparison,
    save_visualization
)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--data-dir',
                      type=str,
                      required=True,
                      help='Path to preprocessed test data directory')
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
    parser.add_argument('--batch-size',
                      type=int,
                      default=4,
                      help='Batch size')
    return parser.parse_args()

def load_test_data(data_dir):
    """Load test data."""
    modalities = []
    segmentations = []
    case_ids = []
    
    for file in os.listdir(data_dir):
        if file.endswith('_modalities.npy'):
            case_id = file.replace('_modalities.npy', '')
            mod = np.load(os.path.join(data_dir, file))
            seg = np.load(os.path.join(data_dir, f'{case_id}_segmentation.npy'))
            modalities.append(mod)
            segmentations.append(seg)
            case_ids.append(case_id)
    
    return np.stack(modalities), np.stack(segmentations), case_ids

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = get_model(device=args.device)
    model, _ = load_model(args.model_path, model)
    model.eval()
    
    # Load test data
    print('Loading test data...')
    X_test, y_test, case_ids = load_test_data(args.data_dir)
    
    # Evaluate each case
    results = {}
    
    print('Evaluating...')
    with torch.no_grad():
        for i, (x, y, case_id) in enumerate(tqdm(zip(X_test, y_test, case_ids))):
            # Prepare input
            x = torch.from_numpy(x).unsqueeze(0).to(args.device).float()
            
            # Get prediction
            pred = model(x)
            pred = torch.argmax(pred, dim=1).cpu().numpy()[0]
            
            # Calculate metrics
            metrics = evaluate_prediction(y, pred)
            results[case_id] = metrics
            
            # Save visualization
            fig = plt.figure(figsize=(15, 5))
            plot_prediction_comparison(x[0,0].cpu().numpy(), y, pred)
            save_visualization(fig, output_dir / f'{case_id}_prediction.png')
            plt.close()
    
    # Calculate and save mean metrics
    mean_metrics = {
        'mean_dice': np.mean([r['mean_dice'] for r in results.values()]),
        'mean_iou': np.mean([r['mean_iou'] for r in results.values()]),
        'mean_precision': np.mean([r['mean_precision'] for r in results.values()]),
        'mean_recall': np.mean([r['mean_recall'] for r in results.values()])
    }
    
    results['mean'] = mean_metrics
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print('\nEvaluation Results:')
    print(f"Mean Dice Score: {mean_metrics['mean_dice']:.4f}")
    print(f"Mean IoU Score: {mean_metrics['mean_iou']:.4f}")
    print(f"Mean Precision: {mean_metrics['mean_precision']:.4f}")
    print(f"Mean Recall: {mean_metrics['mean_recall']:.4f}")

if __name__ == '__main__':
    main() 