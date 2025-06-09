"""
Data loading and preprocessing utilities for BraTS2021 dataset.
"""
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
from pathlib import Path

def load_nifti(file_path: str) -> np.ndarray:
    """Load a NIfTI file and return its data as a numpy array."""
    return nib.load(file_path).get_fdata()

def preprocess_scan(scan: np.ndarray,
                   normalize: bool = True,
                   clip_values: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Preprocess a single MRI scan.
    
    Args:
        scan: Input 3D scan
        normalize: Whether to normalize to [0,1] range
        clip_values: Optional tuple of (min, max) values for clipping
    
    Returns:
        Preprocessed scan
    """
    if clip_values:
        scan = np.clip(scan, clip_values[0], clip_values[1])
    
    if normalize:
        scan = (scan - scan.min()) / (scan.max() - scan.min() + 1e-8)
    
    return scan

def load_brats_case(data_dir: str, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single BraTS case including all modalities and segmentation.
    
    Args:
        data_dir: Path to BraTS2021 data directory
        case_id: Case identifier
        
    Returns:
        Tuple of (modalities, segmentation) where modalities is (4, H, W, D)
    """
    modality_files = {
        0: f"{case_id}_flair.nii.gz",
        1: f"{case_id}_t1.nii.gz", 
        2: f"{case_id}_t1ce.nii.gz",
        3: f"{case_id}_t2.nii.gz"
    }
    
    modalities = []
    for idx in range(4):
        path = os.path.join(data_dir, case_id, modality_files[idx])
        scan = load_nifti(path)
        scan = preprocess_scan(scan)
        modalities.append(scan)
    
    seg_path = os.path.join(data_dir, case_id, f"{case_id}_seg.nii.gz")
    segmentation = load_nifti(seg_path)
    
    return np.stack(modalities), segmentation

def get_case_ids(data_dir: str) -> List[str]:
    """Get list of case IDs from the data directory."""
    return [d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))]

def save_preprocessed(save_dir: str,
                     case_id: str,
                     modalities: np.ndarray,
                     segmentation: np.ndarray) -> None:
    """Save preprocessed data to numpy files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / f"{case_id}_modalities.npy", modalities)
    np.save(save_dir / f"{case_id}_segmentation.npy", segmentation)

class BraTSDataset(Dataset):
    """PyTorch Dataset for BraTS data."""
    def __init__(self, data_dir: str, case_ids: List[str]):
        self.data_dir = Path(data_dir)
        self.case_ids = case_ids
        
    def __len__(self) -> int:
        return len(self.case_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_id = self.case_ids[idx]
        
        # Load preprocessed data
        modalities = np.load(self.data_dir / f"{case_id}_modalities.npy")
        segmentation = np.load(self.data_dir / f"{case_id}_segmentation.npy")
        
        # Convert to torch tensors
        modalities = torch.from_numpy(modalities).float()
        segmentation = torch.from_numpy(segmentation).long().unsqueeze(0)
        
        return modalities, segmentation

def load_data(data_dir: str,
             batch_size: int = 2,
             train_val_split: float = 0.8,
             num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation/test data loaders.
    
    Args:
        data_dir: Directory with preprocessed data
        batch_size: Batch size for training
        train_val_split: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get all case IDs
    case_ids = [f.stem.replace('_modalities', '') 
                for f in Path(data_dir).glob('*_modalities.npy')]
    
    # Split into train and val
    np.random.shuffle(case_ids)
    split_idx = int(len(case_ids) * train_val_split)
    train_ids = case_ids[:split_idx]
    val_ids = case_ids[split_idx:]
    
    # Create datasets
    train_dataset = BraTSDataset(data_dir, train_ids)
    val_dataset = BraTSDataset(data_dir, val_ids)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 