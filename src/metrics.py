"""
Evaluation metrics for brain tumor segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List

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

def evaluate_prediction(pred: np.ndarray,
                      target: np.ndarray,
                      threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate segmentation prediction.
    
    Args:
        pred: Predicted segmentation probabilities (C, H, W, D)
        target: Ground truth segmentation (1, H, W, D)
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Debug shapes
    print(f"\nEvaluation shapes:")
    print(f"  pred shape: {pred.shape}")
    print(f"  target shape: {target.shape}")
    
    # Convert predictions to binary
    pred_binary = (pred > threshold).astype(np.float32)
    
    # Convert target to one-hot encoding
    target = target[0]  # Remove batch dimension
    target_onehot = np.zeros((pred.shape[0],) + target.shape, dtype=np.float32)
    for i in range(pred.shape[0]):
        target_onehot[i] = (target == i).astype(np.float32)
    
    print(f"  target_onehot shape: {target_onehot.shape}")
    
    # Calculate metrics for each class
    metrics = {}
    for class_idx in range(pred.shape[0]):
        pred_class = pred_binary[class_idx]
        target_class = target_onehot[class_idx]
        
        # Dice score
        intersection = np.sum(pred_class * target_class)
        union = np.sum(pred_class) + np.sum(target_class)
        dice = (2.0 * intersection) / (union + 1e-5)
        
        # IoU (Jaccard)
        iou = intersection / (union - intersection + 1e-5)
        
        # Precision and recall
        true_positives = np.sum(pred_class * target_class)
        pred_positives = np.sum(pred_class)
        actual_positives = np.sum(target_class)
        
        precision = true_positives / (pred_positives + 1e-5)
        recall = true_positives / (actual_positives + 1e-5)
        
        metrics[f'dice_class_{class_idx}'] = float(dice)
        metrics[f'iou_class_{class_idx}'] = float(iou)
        metrics[f'precision_class_{class_idx}'] = float(precision)
        metrics[f'recall_class_{class_idx}'] = float(recall)
    
    # Calculate mean metrics
    metrics['mean_dice'] = np.mean([v for k, v in metrics.items() if k.startswith('dice')])
    metrics['mean_iou'] = np.mean([v for k, v in metrics.items() if k.startswith('iou')])
    metrics['mean_precision'] = np.mean([v for k, v in metrics.items() if k.startswith('precision')])
    metrics['mean_recall'] = np.mean([v for k, v in metrics.items() if k.startswith('recall')])
    
    return metrics

class DiceLoss(nn.Module):
    """Dice loss for segmentation with class weights."""
    def __init__(self, smooth: float = 1e-5, class_weights: List[float] = None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]) if class_weights is None else torch.tensor(class_weights)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted Dice loss.
        
        Args:
            pred: Predicted segmentation (B, C, H, W, D)
            target: Ground truth segmentation (B, C, H, W, D)
            
        Returns:
            Weighted Dice loss value
        """
        # Debug shapes
        print("Loss calculation shapes:")
        print(f"  pred shape: {pred.shape}")
        print(f"  target shape: {target.shape}")
        
        # Ensure float tensors
        pred = pred.float()
        target = target.float()
        
        # Move class weights to pred device
        self.class_weights = self.class_weights.to(pred.device)
        
        # Calculate Dice loss for each class
        total_loss = 0
        n_classes = pred.shape[1]
        
        for i in range(n_classes):
            pred_class = pred[:, i]
            target_class = target[:, i]
            
            # Flatten predictions and targets
            pred_flat = pred_class.reshape(-1)
            target_flat = target_class.reshape(-1)
            
            # Calculate intersection and union
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            
            # Calculate Dice coefficient
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            
            # Apply class weight
            weighted_dice = self.class_weights[i] * (1.0 - dice)
            total_loss += weighted_dice
        
        return total_loss / n_classes 