# AI-Powered-Brain-Tumor-Segmentation-in-MRI-Scans

# Brain Tumor Segmentation with BraTS2021

## Overview  
This Jupyter notebook implements a complete pipeline for brain tumor segmentation using the BraTS 2021 dataset. The notebook walks you through data extraction, preprocessing, exploration, model‐ready data generation, a ResNet-based segmentation architecture, and evaluation of segmentation metrics (IOU, Dice, precision, recall, specificity, etc.).

## Contents  
1. **Setup & Imports**  
   - Load standard imaging and ML libraries (nilearn, nibabel, scikit-image, TensorFlow, PyTorch, Hugging Face transformers, etc.)
   - Define file paths and environment variables  
2. **Data Extraction**  
   - Unpack the BraTS 2021 training archive and sample patient tarballs  
   - Verify integrity of `.nii`/`.nii.gz` files  
3. **Data Loading & Visualization**  
   - Load MRI modalities (FLAIR, T1, T1ce, T2) and ground-truth segmentation masks  
   - Display representative axial slices side-by-side  
4. **Preprocessing Utilities**  
   - `load_nifti()`, `save_nifti()` for I/O  
   - `preprocess_nifti()` for brightness adjustment & resizing  
   - `show_slices()` to compare raw vs. preprocessed  
5. **Data Generator**  
   - `DataGenerator` class (inherits `tf.keras.utils.Sequence`)  
   - Multi-slice batching, volume-to-slice splitting, on-the-fly resizing, one-hot mask encoding  
   - Creation of training, validation, and test generators  
6. **Model Definition**  
   - Load a backbone ResNet model from Hugging Face (`ResNetModel`)  
   - Wrap it in a `ResNetSegmentation` head for pixel-wise classification  
   - Device setup (CPU/GPU)  
7. **Evaluation & Metrics**  
   - Batch-wise inference via `predict_resnet_eval()`  
   - Compute IoU, Dice (overall and per-tumor-class), precision, recall (sensitivity), specificity  
   - Print summary metrics  
8. **Prediction Visualization**  
   - `predict_single()` and `show_predictions()` to overlay model outputs on slices  
   - Qualitative inspection of segmentation performance  

## Prerequisites  
- Python 3.7+  
- [nilearn](https://nilearn.github.io/)  
- [nibabel](https://nipy.org/nibabel/)  
- [scikit-image](https://scikit-image.org/)  
- [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/)  
- [matplotlib](https://matplotlib.org/)  
- [tensorflow](https://www.tensorflow.org/)  
- [torch](https://pytorch.org/)  
- [transformers](https://huggingface.co/docs/transformers/)  
- [opencv-python](https://pypi.org/project/opencv-python/)  

Install via:

```bash
pip install nilearn nibabel scikit-image numpy pandas matplotlib tensorflow torch transformers opencv-python
