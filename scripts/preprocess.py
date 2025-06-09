#!/usr/bin/env python3
"""
Preprocess BraTS2021 dataset.
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    get_case_ids,
    load_brats_case,
    save_preprocessed
)

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess BraTS2021 dataset')
    parser.add_argument('--data-dir',
                      type=str,
                      required=True,
                      help='Path to BraTS2021 data directory')
    parser.add_argument('--output-dir',
                      type=str,
                      required=True,
                      help='Output directory for preprocessed data')
    parser.add_argument('--num-cases',
                      type=int,
                      default=None,
                      help='Number of cases to process (None for all)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of cases
    cases = get_case_ids(args.data_dir)
    if args.num_cases:
        cases = cases[:args.num_cases]
    
    print(f'Processing {len(cases)} cases...')
    
    # Process each case
    for case_id in tqdm(cases):
        try:
            # Load and preprocess case
            modalities, segmentation = load_brats_case(args.data_dir, case_id)
            
            # Save preprocessed data
            save_preprocessed(output_dir, case_id, modalities, segmentation)
            
        except Exception as e:
            print(f'Error processing case {case_id}: {str(e)}')
            continue
    
    print('Done!')

if __name__ == '__main__':
    main() 