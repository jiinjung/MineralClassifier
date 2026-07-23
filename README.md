# MineralClassifier and Rock Type Classification

[**Cite**] Jung, J. I., Tikoo, S. M., Chung, J., & O. Nichols, C. I. (2026). Automated Mineral Identification and Rock-Type Classification of Lunar Mare Basalts Using SEM Images. Journal of Geophysical Research: Machine Learning and Computation, 3(4), e2025JH001124. https://doi.org/10.1029/2025JH001124

[**Dataset**] Jung, J. I., Tikoo, S., Chung, J.& Nichols, C. I. O. (2026). Data and Software for Automated Mineral Identification and Rock-Type Classification of Lunar Mare Basalts using SEM Images [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.20698563

<p align="center">
 <img width="700" alt="inputoutput" src="https://github.com/jiinjung/MineralClassifier/assets/87342008/529ccc58-3f72-4dde-b052-9d588ef965f6">
</p>
This current system employs deep learning semantic segmentation models to process the input Scanning Electron Microscopy (SEM) images of lunar mare basalts and corresponding pixel scales, and generated a segmented mineral map as output.


# Overview
This project implements a two-stage pipeline:
 
1. Semantic Segmentation: U-Net based models to segment minerals from SEM Back Scattering Electron (BSE) images
2. Rock Classification: Multiple machine learning classifiers to classify rock types based on modal mineral abundances


# Key Features

## Three U-Net variants that handle pixel scale variability:
- UNet_1: Baseline model without scale conditioning  (loss: cross-entropy; CE+Dice and CE+GDL variants)
- UNet_2: Dual UNet with routing based on pixel scale threshold (1.8 μm)
- UNet_3: Continuous scale conditioning at bottleneck layer
 
## Multiple classification approaches:
- Rule-based baseline classifier
- Gaussian Naive Bayes
- Support Vector Machine (RBF)
- Logistic Regression
- Random Forest
- Multilayer Perceptron
- XGBoost

  
# Dataset
The project uses two main datasets:

## 1. Segmentation Dataset (`/data-sem-label`)
Contains SEM images and segmentation masks:
- Input images: 256×256 grayscale SEM images
- Output images: 256×256 segmentation masks with 10 mineral classes
- Pixel scale files: Text files containing pixel size information (0.02-20 μm)
- Structure: Organized in folders with `input-images/`, `output-images/`, and `input-features/` subdirectories
- Mineral classes:
- `c₀`: Void
- `c₁`: Metallic Fe (Kamacite, Martensite)
- `c₂`: FeS (Troilite)
- `c₃`: Metal oxides (Ilmenite, Ulvospinel, Chromite)
- `c₄`: Pyroxene group
- `c₅`: Plagioclase
- `c₆`: Silicate minerals (Quartz, Cristobalite, Glass)
- `c₇`: Olivine
- `c₈`: Late-stage melt compositions (Mesostasis, Groundmass)
- `c₉`: Other (Phosphate, Zircon, etc.)
  
##  2. Classification Dataset (`data-lsc-modal.xlsx`)
Excel file containing modal mineral abundance data for rock classification:
Mineral abundance columns:
- `Ol`: Olivine abundance
- `Py`: Pyroxene abundance
- `Pl`: Plagioclase abundance
- `Ms`: Mesostasis abundance
- `Si`: Silica minerals abundance
- `Op`: Opaque minerals abundance (Fe-Ni, FeS, metal oxides)
Rock type labels: 
- Ilmenite basalt (class 0)
- Olivine basalt (class 1)
- Pigeonite basalt (class 2)
 
- Used for training and evaluating rock classification models based on modal mineral abundances extracted from segmentation results
# Training Details
## Segmentation Models
- U-Net with encoder-decoder structure and skip connections
- Input: 256×256 grayscale images (+ pixel scale (scalar))
- Output: 256×256 segmentation masks (10 classes)
- Loss: Cross-entropy loss, Dice loss, Generalized dice loss
- Metric: Pixel accuracy, intersection over union
- Data augmentation: 90-degree rotations, brightness adjustment
 
## Classification Models
### Features: Modal mineral abundances extracted from segmentation masks:
- `Ol`: Olivine abundance
- `Py`: Pyroxene abundance
- `Pl`: Plagioclase abundance
- `Ms`: Mesostasis abundance
- `Si`: Silicate minerals abundance
- `Op`: Opaque minerals abundance (Fe-Ni, FeS, metal oxides)
- Classes: 3 rock types (Ilmenite basalt, Olivine basalt, Pigeonite basalt)

# Workflow:
1. Segmentation model extracts mineral pixel counts from SEM images
2. Pixel counts are converted to modal abundances (percentages)
3. Classification models predict rock type from modal abundances




