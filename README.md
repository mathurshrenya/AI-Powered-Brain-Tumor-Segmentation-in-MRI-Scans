# AI-Powered Brain Tumor Segmentation in MRI Scans

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-orange)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🧠 Brain Tumor Segmentation with BraTS2021

### Overview
This project implements a state-of-the-art deep learning pipeline for automated brain tumor segmentation using the BraTS 2021 dataset. The system utilizes advanced neural network architectures to accurately identify and segment brain tumors from multiple MRI modalities.

![Project Banner](https://raw.githubusercontent.com/username/AI-Powered-Brain-Tumor-Segmentation-in-MRI-Scans/main/assets/banner.png)

### 🌟 Key Features
- Multi-modal MRI processing (FLAIR, T1, T1ce, T2)
- ResNet-based segmentation architecture
- Comprehensive evaluation metrics
- Interactive Jupyter notebook implementation
- Automated preprocessing pipeline
- Visualization tools for results analysis

### 📊 Project Structure
```
.
├── brats-dataset-segmentation1.ipynb  # Main implementation notebook
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation
├── data/                             # Dataset directory (not tracked)
│   ├── raw/                          # Raw BRATS2021 data
│   └── processed/                    # Preprocessed data
├── models/                           # Saved model checkpoints
└── results/                          # Output visualizations
```

### 🔧 Contents
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

### 🚀 Getting Started

1. **Clone the repository**
```bash
git clone https://github.com/username/AI-Powered-Brain-Tumor-Segmentation-in-MRI-Scans.git
cd AI-Powered-Brain-Tumor-Segmentation-in-MRI-Scans
```

2. **Set up the environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Download the data**
- Register at [BRATS 2021](https://www.med.upenn.edu/cbica/brats2021/)
- Download and place the dataset in the `data/raw` directory

4. **Run the notebook**
```bash
jupyter notebook brats-dataset-segmentation1.ipynb
```

### 📈 Results
- Average Dice Score: X.XX
- Mean IoU: X.XX
- Precision: X.XX
- Recall: X.XX

[Add visualization of results here]

### 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments
- BraTS 2021 dataset providers
- The medical imaging research community
- Contributors and maintainers of the used libraries

### 📧 Contact
Your Name - [@your_twitter](https://twitter.com/your_twitter)

Project Link: [https://github.com/username/AI-Powered-Brain-Tumor-Segmentation-in-MRI-Scans](https://github.com/username/AI-Powered-Brain-Tumor-Segmentation-in-MRI-Scans)

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
