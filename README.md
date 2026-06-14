# Explainable 3D-CNN for Schizophrenia Detection using Multi-Site Structural MRI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)]()
[![MRI](https://img.shields.io/badge/Application-Neuroimaging-green.svg)]()
[![Research](https://img.shields.io/badge/Research-Schizophrenia%20Detection-orange.svg)]()

## Overview

This repository contains the implementation of an **Explainable 3D Convolutional Neural Network (3D-CNN)** for automated schizophrenia detection using structural MRI (sMRI) data. The framework combines volumetric deep learning with explainable AI techniques to provide both accurate classification and clinically meaningful interpretation of model predictions.

The proposed model was developed and evaluated on the **COINSTAC MCIC VBM multi-site structural MRI dataset** and incorporates **Grad-CAM-based explainability** with quantitative anatomical validation using the **Automated Anatomical Labeling (AAL) Atlas**.

The study demonstrates that volumetric deep learning combined with quantitative interpretability can serve as a robust and biologically meaningful approach for schizophrenia classification.

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
- Grad-CAM based explainability
- AAL Atlas based anatomical validation
- ROI Coverage analysis
- Statistical significance testing using permutation testing
- Quantitative performance evaluation using Accuracy, F1 Score, Cohen's Kappa, and AUC
- Clinically interpretable schizophrenia detection

---

## Dataset

The model was trained and evaluated using the **COINSTAC MCIC VBM Dataset**, a large multi-site structural MRI dataset.

### Dataset Characteristics

| Attribute | Value |
|------------|---------|
| Total Subjects | 3,729 |
| Imaging Modality | Structural MRI (VBM) |
| Classification Task | Binary |
| Classes | Control, Schizophrenia |
| Data Source | COINSTAC MCIC |

The dataset includes MRI scans from both healthy controls and schizophrenia patients acquired across multiple imaging sites, improving model robustness and generalization.

---

## Methodology

### MRI Preprocessing

Each MRI volume undergoes the following preprocessing steps:

1. Loading NIfTI MRI volumes
2. Handling missing values
3. Z-score intensity normalization
4. Spatial resampling
5. Tensor conversion for deep learning

All MRI scans are resampled to a common volumetric resolution:

```text
64 × 64 × 64 voxels
```

This ensures consistent input dimensions across subjects.

---

## Model Architecture

The proposed architecture is a lightweight 3D Convolutional Neural Network designed for volumetric brain MRI analysis.

### Network Structure

```text
Input MRI Volume
(1 × 64 × 64 × 64)

│
├── Conv3D (1 → 16)
├── BatchNorm3D
├── ReLU
├── MaxPool3D

│
├── Conv3D (16 → 32)
├── BatchNorm3D
├── ReLU
├── MaxPool3D

│
├── Conv3D (32 → 64)
├── BatchNorm3D
├── ReLU
├── MaxPool3D

│
├── Flatten

│
├── Fully Connected (32768 → 128)
├── ReLU
├── Dropout (0.5)

│
└── Fully Connected (128 → 2)

Output:
Control / Schizophrenia
```

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 1e-5 |
| Loss Function | CrossEntropy Loss |
| Batch Size | 4 |
| Epochs | 40 |
| Input Size | 64×64×64 |
| Dropout Rate | 0.5 |
| Train-Test Split | 80:20 |

---

## Explainability Framework

Medical AI systems require transparency and interpretability. To address this, the framework integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

### Grad-CAM Workflow

```text
MRI Volume
      │
      ▼
3D CNN Prediction
      │
      ▼
Gradient Extraction
      │
      ▼
Feature Map Weighting
      │
      ▼
3D Activation Map
      │
      ▼
MRI Overlay Visualization
```

Grad-CAM identifies the brain regions that contribute most strongly to the model's classification decision, providing insight into the learned neuroanatomical patterns.

---

## Anatomical Validation

To quantitatively evaluate the biological relevance of model attention, Grad-CAM activation maps are compared against anatomically defined regions from the **AAL Atlas**.

### Regions of Interest (ROIs)

- Hippocampus
- Superior Temporal Gyrus (STG)
- Thalamus
- Frontal Cortex

---

## Explainability Evaluation

### ROI Coverage

Coverage is calculated as:

```text
Coverage = Activated Voxels inside ROI
           --------------------------
             Total Voxels in ROI
```

This measures the proportion of each anatomical region highlighted by Grad-CAM.

### Permutation Testing

A 500-iteration permutation test is performed to determine whether observed ROI coverage values are significantly greater than chance.

The procedure:

1. Randomly shuffle activation masks.
2. Preserve total voxel counts.
3. Generate an empirical null distribution.
4. Compare observed coverage against random coverage.
5. Compute statistical significance.

---

## Results

### Classification Performance

| Metric | Score |
|----------|----------|
| Accuracy | 92.49% |
| F1 Score | 0.9169 |
| Cohen's Kappa | 0.8486 |
| AUC | 0.9886 |

### Test Set Performance

```text
Correct Predictions : 690
Total Samples       : 746
Test Accuracy       : 92.49%
```

---

## Explainability Results

Grad-CAM consistently highlighted neuroanatomical regions previously associated with schizophrenia in neuroimaging literature.

### ROI Coverage Results

| Brain Region | Coverage |
|--------------|----------|
| Superior Temporal Gyrus (STG) | 85.04% |
| Thalamus | 52.32% |
| Frontal Cortex | 45.92% |
| Hippocampus | 22.96% |

### Statistical Significance

Permutation testing demonstrated that all reported ROI coverage values were significantly greater than chance, indicating that model attention is not spatially random and aligns with clinically relevant neuroanatomical structures.

---

## Experimental Pipeline

```text
Structural MRI Data
          │
          ▼
Preprocessing
(Normalization + Resampling)
          │
          ▼
3D CNN Training
          │
          ▼
Schizophrenia Classification
          │
          ▼
Grad-CAM Generation
          │
          ▼
AAL Atlas Validation
          │
          ├── ROI Coverage Analysis
          │
          └── Permutation Testing
```

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

Or create a requirements file and run:

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

### Step 1: Configure Dataset Path

```python
BASE_DIR = "path/to/coinstac_dataset"
```

### Step 2: Configure AAL Atlas Path

```python
AAL_PATH = "path/to/AAL_atlas.nii"
```

### Step 3: Run the Notebook

```bash
jupyter notebook Schiz_Review.ipynb
```

Alternatively, execute the Python script directly.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Nibabel
- Nilearn
- SciPy
- Scikit-Learn
- Matplotlib
- Google Colab

---

## Applications

This framework can be extended to:

- Neuropsychiatric disorder classification
- Alzheimer's disease detection
- Parkinson's disease analysis
- Brain tumor characterization
- Explainable medical imaging AI systems

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{raza2026explainable3dcnn,
  title={Explainable 3D-CNN for Schizophrenia Detection using Multi-Site Structural MRI},
  author={Raza, Sameen and Gupta, Amogh and Mishra, Subhashree and Mishra, Bhabani Shankar Prasad},
  year={2026}
}
```

---

## Acknowledgements

- KIIT University
- COINSTAC Consortium
- Mind Clinical Imaging Consortium (MCIC)
- PyTorch Community
- Nilearn Developers
- Nibabel Developers

---

## License

This project is intended for academic and research purposes.

Please ensure compliance with the licensing and usage restrictions of the COINSTAC MCIC dataset before redistribution or commercial use.
