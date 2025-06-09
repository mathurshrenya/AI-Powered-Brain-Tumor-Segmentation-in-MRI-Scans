"""
Visualization utilities for brain tumor segmentation.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def plot_segmentation(image: np.ndarray,
                     true_mask: np.ndarray,
                     pred_mask: np.ndarray,
                     alpha: float = 0.3) -> plt.Figure:
    """
    Plot image with true and predicted segmentation masks.
    
    Args:
        image: Input MRI slice (H, W)
        true_mask: Ground truth segmentation mask (H, W)
        pred_mask: Predicted segmentation mask (H, W)
        alpha: Transparency for segmentation overlay
        
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('MRI Scan')
    ax1.axis('off')
    
    # Plot true segmentation
    ax2.imshow(image, cmap='gray')
    mask = np.ma.masked_where(true_mask == 0, true_mask)
    ax2.imshow(mask, cmap='Set1', alpha=alpha)
    ax2.set_title('True Segmentation')
    ax2.axis('off')
    
    # Plot predicted segmentation
    ax3.imshow(image, cmap='gray')
    mask = np.ma.masked_where(pred_mask == 0, pred_mask)
    ax3.imshow(mask, cmap='Set1', alpha=alpha)
    ax3.set_title('Predicted Segmentation')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_progress(train_losses: list,
                         val_losses: list,
                         metrics: dict,
                         save_path: str) -> None:
    """
    Plot training progress.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    metric_names = list(metrics.keys())
    metric_values = [metrics[name] for name in metric_names]
    ax2.bar(metric_names, metric_values)
    ax2.set_title('Evaluation Metrics')
    ax2.set_ylabel('Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_montage(images: list,
                  masks: list,
                  n_rows: int,
                  n_cols: int,
                  figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create a montage of images with segmentation masks.
    
    Args:
        images: List of input images
        masks: List of segmentation masks
        n_rows: Number of rows in montage
        n_cols: Number of columns in montage
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()
    
    for idx, (img, mask) in enumerate(zip(images, masks)):
        if idx < len(axes):
            axes[idx].imshow(img, cmap='gray')
            mask = np.ma.masked_where(mask == 0, mask)
            axes[idx].imshow(mask, cmap='Set1', alpha=0.3)
            axes[idx].axis('off')
    
    # Turn off any unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig 