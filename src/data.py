"""
Data loading and preprocessing utilities for BraTS2021 dataset.
"""
import os
import numpy as np
import nibabel as nib
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