# Pneumonia X-Ray Prediction

A deep learning project that classifies chest X-ray images as **PNEUMONIA** or **NORMAL** using DNN and CNN models built with TensorFlow/Keras.

## Overview

Pneumonia is a serious lung infection that can be life-threatening, particularly in young children and the elderly. This project automates chest X-ray diagnosis to reduce diagnostic errors, save time, and improve healthcare accessibility in underserved areas.

**Dataset:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
Retrospective cohort of pediatric patients (ages 1–5) from Guangzhou Women and Children's Medical Center.

## Results

| Model | Test Accuracy | Misclassifications |
|-------|-------------|-------------------|
| DNN (baseline) | ~81% | 118 images |
| **CNN (final)** | **~92%** | **55 images** |

## Project Structure

```
Pneumonia_XRAY_Prediction.ipynb
├── 1. Introduction
├── 2. Library Imports
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
pip install tensorflow keras scikit-learn numpy pandas matplotlib seaborn opencv-python kagglehub
```

## Key Steps

### Preprocessing
- Images resized to **150×150**, converted to grayscale
- Pixel values normalized to `[0, 1]`
- Class imbalance addressed via **data augmentation** (rotation, shifts, zoom, horizontal flip)

### Models

**DNN (Baseline)**
```
Flatten → Dense(128, relu) → Dense(64, relu) → Dense(1, sigmoid)
```

**CNN (Final)**
```
Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(1, sigmoid)
```

Both models use:
- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Early stopping on `val_loss` (patience=5)

### EDA
- Class distribution bar chart (dataset is imbalanced — more pneumonia samples)
- t-SNE + PCA visualization confirms class separability with some overlap/outliers

## Reproducibility

Seeds are set across Python, NumPy, TensorFlow, and environment variables (`PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`) for consistent results.

## Conclusion

The CNN achieved **~92% accuracy**, outperforming the DNN baseline by ~11%. CNNs are better suited for image classification due to their ability to detect spatial patterns and hierarchies, leading to fewer misclassifications and reduced overfitting. 
