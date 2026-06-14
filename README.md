# Explainable 3D-CNN for Schizophrenia Detection using Multi-Site Structural MRI

## Overview

This repository contains the implementation of an **Explainable 3D Convolutional Neural Network (3D-CNN)** for automated schizophrenia classification using structural MRI (sMRI) data. The framework combines volumetric deep learning with **Grad-CAM-based explainability** and quantitative anatomical validation to provide clinically interpretable predictions.

The model was developed and evaluated on the **COINSTAC MCIC VBM multi-site dataset**, consisting of MRI scans collected across multiple acquisition sites. In addition to classification performance, the model identifies brain regions contributing to its predictions and validates them against established neuroanatomical regions associated with schizophrenia.

---

## Research Paper

**Title:** Explainable 3D-CNN for Schizophrenia Detection using Multi-Site Structural MRI

**Authors**
- Sameen Raza
- Amogh Gupta
- Subhashree Mishra
- Bhabani Shankar Prasad Mishra

---

## Key Features

- End-to-end 3D CNN for volumetric MRI classification
- Multi-site structural MRI analysis
- Automated preprocessing pipeline
- Grad-CAM explainability for model interpretation
- Atlas-based ROI validation using the AAL Atlas
- Dice-score analysis for anatomical overlap
- ROI coverage analysis
- Permutation testing for statistical significance
- Quantitative evaluation using Accuracy, F1 Score, and AUC

---

## Dataset

This work uses the **COINSTAC MCIC VBM Dataset**, a multi-site structural MRI dataset containing scans from:

- Healthy Controls
- Schizophrenia Patients

### Dataset Statistics

| Attribute | Value |
|------------|---------|
| Total Subjects | 3,729 |
| Data Type | Structural MRI (VBM) |
| Task | Binary Classification |
| Classes | Control, Schizophrenia |

---

## Methodology

### MRI Preprocessing

Each MRI volume undergoes:

1. Loading from NIfTI format
2. NaN value removal
3. Z-score normalization
4. Resampling to a fixed resolution of:

```text
64 × 64 × 64 voxels
```

5. Conversion to PyTorch tensors

---

### Model Architecture

Input Volume:

```text
(1, 64, 64, 64)
```

Architecture:

```text
Conv3D (1 → 16)
│
├── BatchNorm3D
├── ReLU
└── MaxPool3D

Conv3D (16 → 32)
│
├── BatchNorm3D
├── ReLU
└── MaxPool3D

Conv3D (32 → 64)
│
├── BatchNorm3D
├── ReLU
└── MaxPool3D

Flatten

Linear (32768 → 128)
│
├── ReLU
└── Dropout(0.5)

Linear (128 → 2)
```

Output Classes:

- Control
- Schizophrenia

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 1e-5 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 4 |
| Epochs | 40 |
| Input Size | 64×64×64 |
| Dropout | 0.5 |

---

## Explainability Pipeline

To improve clinical interpretability, the framework integrates **3D Grad-CAM**.

### Workflow

1. Extract feature maps from the final convolutional layer.
2. Compute gradients with respect to the predicted class.
3. Generate class activation maps.
4. Upsample activations to MRI resolution.
5. Visualize regions driving model predictions.

---

## Anatomical Validation

The generated Grad-CAM maps are quantitatively validated using the **Automated Anatomical Labeling (AAL) Atlas**.

### Regions of Interest (ROIs)

- Hippocampus
- Superior Temporal Gyrus (STG)
- Thalamus
- Frontal Cortex

### Evaluation Metrics

#### Dice Similarity Coefficient

Measures overlap between Grad-CAM activation regions and anatomical ROIs.

#### ROI Coverage

Measures the percentage of ROI voxels covered by model attention.

#### Permutation Testing

500 random permutations are used to evaluate whether the observed attention patterns are statistically significant.

---

## Results

### Classification Performance

| Metric | Score |
|----------|----------|
| Accuracy | 92.49% |
| F1 Score | 0.9169 |
| Cohen's Kappa | 0.8486 |
| AUC | 0.9886 |

### Test Performance

```text
Correct Predictions: 690 / 746
Test Accuracy: 92.49%
```

---

## Explainability Results

The model consistently focused on neuroanatomical regions known to be associated with schizophrenia.

### ROI Coverage

| Brain Region | Coverage |
|--------------|----------|
| Superior Temporal Gyrus | 85.04% |
| Thalamus | 52.32% |
| Frontal Cortex | 45.92% |
| Hippocampus | 22.96% |

### Statistical Validation

All major ROI activations demonstrated significantly greater-than-random overlap during permutation testing, supporting the neurobiological relevance of model attention.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git

cd <repository-name>
```

### Install Dependencies

```bash
pip install torch torchvision
pip install nibabel
pip install nilearn
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install tqdm
```

Or:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
.
├── Schiz_Review.ipynb
├── README.md
├── requirements.txt
├── figures/
│   ├── architecture.png
│   ├── gradcam_visualization.png
│   ├── roc_curve.png
│   └── training_curves.png
├── paper/
│   └── Explainable_3D_CNN_Schizophrenia.pdf
└── dataset/
```

---

## Running the Project

Update the dataset path:

```python
BASE_DIR = "path/to/coinstac_dataset"
```

Update the AAL atlas path:

```python
AAL_PATH = "path/to/AAL_atlas.nii"
```

Run the notebook:

```bash
jupyter notebook Schiz_Review.ipynb
```

Or execute as a Python script.

---

## Experimental Pipeline

```text
MRI Data
   │
   ▼
Preprocessing
(Normalization + Resampling)
   │
   ▼
3D CNN Training
   │
   ▼
Classification
(Control vs Schizophrenia)
   │
   ▼
Grad-CAM Explainability
   │
   ▼
AAL Atlas Validation
   │
   ├── Dice Score
   ├── ROI Coverage
   └── Permutation Testing
```

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Nibabel
- Nilearn
- SciPy
- Scikit-learn
- Matplotlib

---

## Future Work

- Cross-site external validation
- Multimodal MRI integration
- Federated learning deployment
- Transformer-based volumetric architectures
- Advanced explainability frameworks

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{raza2026explainable3dcnn,
  title={Explainable 3D-CNN for Schizophrenia Detection using Multi-Site Structural MRI},
  author={Raza, Sameen and Gupta, Amogh and Mishra, Subhashree and Mishra, Bhabani Shankar Prasad},
  year={2026}
}
```

---

## License

This repository is intended for academic and research purposes.

Please ensure compliance with the licensing and usage restrictions of the COINSTAC MCIC dataset before redistribution or commercial use.

---

## Acknowledgements

- KIIT University
- COINSTAC Consortium
- Mind Clinical Imaging Consortium (MCIC)
- PyTorch Community
- Nilearn and Nibabel Developers
