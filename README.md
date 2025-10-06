# Schizophrenia-detection-using-3D-CNN-Grad-CAM
This project develops an end-to-end deep learning framework to classify structural MRI (sMRI) scans into schizophrenia and healthy control groups using 3D CNNs. It automatically learns spatial brain features from volumetric data and provides interpretable insights into brain regions linked to schizophrenia.
# 🚀 Key Features

Preprocessing of MRI volumes using Nibabel and PyTorch

Standardization to zero mean and unit variance

Uniform resizing of all scans to 64×64×64 voxels

Custom MRIDataset class for efficient data loading

3D CNN architecture with Conv3D–BatchNorm–ReLU–MaxPool blocks

Adam optimizer and Cross-Entropy Loss for training

Model evaluation based on test accuracy

Grad-CAM explainability to highlight discriminative brain regions

# 🧩 Methodology

## Data Collection & Preprocessing

MRI scans and metadata obtained from multiple research sites.

Volumes normalized, resized, and converted to tensors.

## Dataset Splitting

80% training and 20% testing subsets.

## Model Architecture

Three convolutional layers followed by two fully connected layers for binary classification.

## Training Setup

Optimizer: Adam

Loss Function: Cross-Entropy

Evaluation Metric: Accuracy

## Explainability

Grad-CAM applied to visualize brain regions influencing predictions.

# 💻 Technologies Used

Python 3.10+

PyTorch

Nibabel

NumPy, Pandas, Matplotlib

Google Colab for training and visualization

# 🧠 Results & Interpretability

The model successfully differentiates between schizophrenia and healthy control MRI scans. Grad-CAM heatmaps reveal key brain regions influencing the classification, improving the model’s transparency and interpretability.

# 👥 Team

This project was developed collaboratively by a two-member research team as part of an academic study on MRI-based schizophrenia classification.
