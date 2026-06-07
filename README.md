# Pneumonia X-Ray Prediction

A deep learning project classifying pediatric chest X-rays as **PNEUMONIA** or **NORMAL** using DNN and CNN models built with TensorFlow/Keras.

## Overview

Diagnosing pneumonia traditionally requires a clinician to manually review chest X-rays — a time-consuming process prone to error. This project explores whether deep learning can automate classification to reduce diagnostic errors and improve healthcare accessibility in underserved areas.

**Dataset:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
Retrospective cohort of pediatric patients (ages 1–5) from Guangzhou Women and Children's Medical Center | 2 classes: `PNEUMONIA`, `NORMAL`

## Results

| Model | Test Accuracy | Misclassifications |
|-------|-------------|-------------------|
| DNN (baseline) | ~81% | 118 images |
| **CNN (final)** | **~92%** | **55 images** |

CNN outperformed the DNN baseline by ~11%, with spatial pattern detection via convolutional layers reducing misclassifications by more than half.

## Project Structure

```
Pneumonia_XRAY_Prediction.ipynb
├── 1. Introduction
├── 2. Importing Libraries
├── 3. Reproducibility Settings
├── 4. Data Loading
├── 5. EDA & Preprocessing
├── 6. t-SNE with PCA Visualization
├── 7. Data Augmentation
├── 8. DNN Model (baseline)
└── 9. CNN Model (final)
```

## Requirements

```bash
pip install tensorflow keras scikit-learn numpy matplotlib seaborn opencv-python kagglehub
```

## Key Steps

### Data Loading
- Images loaded in grayscale, resized to **150×150**
- Split into `train/`, `test/`, `val/` directories via `kagglehub`

### Data Cleaning & Preprocessing
- No null values; pixel values normalized to `[0, 1]`
- Class imbalance (more pneumonia than normal samples) addressed via data augmentation rather than class weighting
- Images reshaped to `(-1, 150, 150, 1)` for model input

### Exploratory Analysis
- Class distribution bar chart confirmed imbalance
- t-SNE + PCA (50 components) on training data showed correct clustering with some overlap/outliers

### Data Augmentation
- Applied to training data only via `ImageDataGenerator`
- Parameters: `rotation_range=30`, `width/height_shift=0.1`, `zoom=0.1`, `horizontal_flip=True`

### Model 1: DNN (Baseline)
- Architecture: `Flatten → Dense(128, relu) → Dense(64, relu) → Dense(1, sigmoid)`
- Optimizer: `Adam` | Loss: `Binary Crossentropy`
- Early stopping on `val_loss` (patience=5), trained for up to 10 epochs

### Model 2: CNN (Final)
- Architecture: `Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(1, sigmoid)`
- Same compile/fit settings as DNN
- Convolutional layers capture spatial hierarchies, improving accuracy and reducing overfitting

## Conclusion

The CNN achieved ~92% accuracy, meeting the >90% target and validating that convolutional layers are better suited for image classification than flat dense layers. The DNN at ~81% served as a useful baseline. Future improvements could include adding Dropout layers, experimenting with transfer learning (e.g., ResNet, VGG), or expanding the dataset to adult patients.
