#!/usr/bin/env python3
"""
Create synthetic test data for testing the segmentation pipeline.
"""
import os
import numpy as np
from pathlib import Path
import nibabel as nib

def create_synthetic_scan(size=(240, 240, 155)):
    """Create a synthetic MRI scan with a simulated tumor."""
    # Create base scan
    scan = np.random.normal(0.5, 0.1, size)
    
    # Add a simulated tumor (bright region)
    center = np.array(size) // 2
    x, y, z = np.ogrid[:size[0], :size[1], :size[2]]
    tumor_mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= 20**2
    scan[tumor_mask] += 0.5
    
    return scan

def create_synthetic_segmentation(size=(240, 240, 155)):
    """Create a synthetic segmentation mask."""
    seg = np.zeros(size)
    
    # Add tumor regions
    center = np.array(size) // 2
    x, y, z = np.ogrid[:size[0], :size[1], :size[2]]
    
    # Necrotic tumor core (label 1)
    core_mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= 10**2
    seg[core_mask] = 1
    
    # Edema (label 2)
    edema_mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= 20**2
    seg[edema_mask & ~core_mask] = 2
    
    # Enhancing tumor (label 3)
    enhancing_mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= 15**2
    seg[enhancing_mask & ~core_mask] = 3
    
    return seg

def save_nifti(data, filename):
    """Save numpy array as NIfTI file."""
    nifti = nib.Nifti1Image(data, np.eye(4))
    nib.save(nifti, filename)

def main():
    # Create directories
    data_dir = Path('data/BraTS2021_Training_Data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 synthetic cases
    for i in range(5):
        case_dir = data_dir / f'BraTS2021_{i:03d}'
        case_dir.mkdir(exist_ok=True)
        
        # Create modalities
        modalities = {
            'flair': create_synthetic_scan(),
            't1': create_synthetic_scan(),
            't1ce': create_synthetic_scan(),
            't2': create_synthetic_scan()
        }
        
        # Create segmentation
        segmentation = create_synthetic_segmentation()
        
        # Save files
        for mod_name, mod_data in modalities.items():
            save_nifti(mod_data, case_dir / f'BraTS2021_{i:03d}_{mod_name}.nii.gz')
        
        save_nifti(segmentation, case_dir / f'BraTS2021_{i:03d}_seg.nii.gz')
    
    print('Created synthetic test data in:', data_dir)

if __name__ == '__main__':
    main() 