"""
Evaluation metrics for brain tumor segmentation.
"""
import numpy as np
import torch
from typing import Tuple, Dict

def dice_coefficient(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    smooth: float = 1e-7) -> float:
    """
    Calculate Dice coefficient.
    
    Args:
        y_true: Ground truth binary segmentation
        y_pred: Predicted binary segmentation
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / \
           (np.sum(y_true) + np.sum(y_pred) + smooth)

def iou_score(y_true: np.ndarray,
             y_pred: np.ndarray,
             smooth: float = 1e-7) -> float:
    """
    Calculate Intersection over Union (IoU) score.
    
    Args:
        y_true: Ground truth binary segmentation
        y_pred: Predicted binary segmentation
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def precision_recall(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    smooth: float = 1e-7) -> Tuple[float, float]:
    """
    Calculate precision and recall.
    
    Args:
        y_true: Ground truth binary segmentation
        y_pred: Predicted binary segmentation
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Tuple of (precision, recall)
    """
    true_positives = np.sum(y_true * y_pred)
    predicted_positives = np.sum(y_pred)
    actual_positives = np.sum(y_true)
    
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    recall = (true_positives + smooth) / (actual_positives + smooth)
    
    return precision, recall

def evaluate_prediction(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      n_classes: int = 4) -> Dict[str, float]:
    """
    Evaluate segmentation prediction with multiple metrics.
    
    Args:
        y_true: Ground truth segmentation (H, W, D)
        y_pred: Predicted segmentation (H, W, D)
        n_classes: Number of classes including background
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calculate metrics for each class
    for c in range(1, n_classes):  # Skip background class
        y_true_c = (y_true == c).astype(np.float32)
        y_pred_c = (y_pred == c).astype(np.float32)
        
        metrics[f'dice_class_{c}'] = dice_coefficient(y_true_c, y_pred_c)
        metrics[f'iou_class_{c}'] = iou_score(y_true_c, y_pred_c)
        precision, recall = precision_recall(y_true_c, y_pred_c)
        metrics[f'precision_class_{c}'] = precision
        metrics[f'recall_class_{c}'] = recall
    
    # Calculate mean metrics
    dice_scores = [metrics[f'dice_class_{c}'] for c in range(1, n_classes)]
    iou_scores = [metrics[f'iou_class_{c}'] for c in range(1, n_classes)]
    precision_scores = [metrics[f'precision_class_{c}'] for c in range(1, n_classes)]
    recall_scores = [metrics[f'recall_class_{c}'] for c in range(1, n_classes)]
    
    metrics['mean_dice'] = np.mean(dice_scores)
    metrics['mean_iou'] = np.mean(iou_scores)
    metrics['mean_precision'] = np.mean(precision_scores)
    metrics['mean_recall'] = np.mean(recall_scores)
    
    return metrics

class DiceLoss(torch.nn.Module):
    """Dice loss for training."""
    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        
    def forward(self,
               predictions: torch.Tensor,
               targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            predictions: Model predictions (B, C, H, W, D)
            targets: Ground truth (B, H, W, D)
            
        Returns:
            Dice loss value
        """
        predictions = torch.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets = torch.nn.functional.one_hot(targets.long(), num_classes=predictions.shape[1])
        targets = targets.permute(0, 4, 1, 2, 3)
        
        # Calculate Dice loss
        intersection = torch.sum(predictions * targets, dim=[2,3,4])
        union = torch.sum(predictions, dim=[2,3,4]) + torch.sum(targets, dim=[2,3,4])
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean() 