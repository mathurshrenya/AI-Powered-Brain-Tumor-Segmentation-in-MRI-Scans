"""
Visualization utilities for MRI scans and segmentation results.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

def plot_scan_with_mask(scan: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       slice_idx: Optional[int] = None,
                       alpha: float = 0.3,
                       cmap: str = 'viridis') -> None:
    """
    Plot a single slice of an MRI scan with optional segmentation mask overlay.
    
    Args:
        scan: 3D MRI scan
        mask: Optional binary segmentation mask
        slice_idx: Slice index (if None, use middle slice)
        alpha: Transparency of the mask overlay
        cmap: Colormap for the mask
    """
    if slice_idx is None:
        slice_idx = scan.shape[-1] // 2
        
    plt.figure(figsize=(10, 5))
    
    # Plot original scan
    plt.subplot(1, 2, 1)
    plt.imshow(scan[:, :, slice_idx], cmap='gray')
    plt.title('Original Scan')
    plt.axis('off')
    
    # Plot with mask overlay if provided
    plt.subplot(1, 2, 2)
    plt.imshow(scan[:, :, slice_idx], cmap='gray')
    if mask is not None:
        plt.imshow(mask[:, :, slice_idx],
                  alpha=alpha,
                  cmap=cmap)
    plt.title('Scan with Segmentation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_modalities(modalities: np.ndarray,
                   slice_idx: Optional[int] = None,
                   titles: Optional[List[str]] = None) -> None:
    """
    Plot different MRI modalities side by side.
    
    Args:
        modalities: Array of shape (n_modalities, H, W, D)
        slice_idx: Slice index (if None, use middle slice)
        titles: List of titles for each modality
    """
    if slice_idx is None:
        slice_idx = modalities.shape[-1] // 2
        
    if titles is None:
        titles = [f'Modality {i+1}' for i in range(modalities.shape[0])]
        
    n_modalities = modalities.shape[0]
    fig, axes = plt.subplots(1, n_modalities, figsize=(4*n_modalities, 4))
    
    for i in range(n_modalities):
        axes[i].imshow(modalities[i, :, :, slice_idx], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_prediction_comparison(scan: np.ndarray,
                            true_mask: np.ndarray,
                            pred_mask: np.ndarray,
                            slice_idx: Optional[int] = None) -> None:
    """
    Plot original scan with ground truth and predicted segmentation masks.
    
    Args:
        scan: 3D MRI scan
        true_mask: Ground truth segmentation mask
        pred_mask: Predicted segmentation mask
        slice_idx: Slice index (if None, use middle slice)
    """
    if slice_idx is None:
        slice_idx = scan.shape[-1] // 2
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original scan
    axes[0].imshow(scan[:, :, slice_idx], cmap='gray')
    axes[0].set_title('Original Scan')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(scan[:, :, slice_idx], cmap='gray')
    axes[1].imshow(true_mask[:, :, slice_idx],
                  alpha=0.3,
                  cmap='viridis')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(scan[:, :, slice_idx], cmap='gray')
    axes[2].imshow(pred_mask[:, :, slice_idx],
                  alpha=0.3,
                  cmap='viridis')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_visualization(fig: plt.Figure,
                     path: str,
                     dpi: int = 300) -> None:
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        path: Save path
        dpi: Resolution
    """
    fig.savefig(path, dpi=dpi, bbox_inches='tight') 