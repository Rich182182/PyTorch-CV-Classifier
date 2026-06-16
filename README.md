# PyTorch Advanced Image Classification & Autoencoders

An end-to-end Deep Learning and Computer Vision pipeline implemented in PyTorch. This project demonstrates a comprehensive approach to image analysis, ranging from building custom Convolutional Neural Networks (CNNs) from scratch to fine-tuning state-of-the-art vision models, applying advanced regularization techniques, and exploring latent spaces using unsupervised Autoencoders.

## 🚀 Project Overview

The primary objective of this project is to classify natural scenes into 6 distinct categories (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`) while optimizing model performance through systematic experimentation. 

**Key Highlights:**
* **Modular Architecture:** Clean separation of concerns with dedicated modules for model definitions, custom augmentations, and training pipelines.
* **Transfer Learning & Fine-tuning:** Evaluated multiple architectures (ResNet18, ResNet50, EfficientNet-B0, MobileNet-V2) using both frozen feature extractors and differential learning rates (unfreezing the `layer4` and `head`).
* **Advanced Regularization:** Implemented and compared Label Smoothing, L2 Weight Decay, Dropout, and complex spatial/color augmentations (Random Affine, Random Erasing, Color Jitter).
* **Unsupervised Learning:** Built Convolutional and Denoising Autoencoders to extract 128-dimensional latent representations, visualized using t-SNE, and evaluated image reconstruction using PSNR.
* **Model Ensembling:** Achieved state-of-the-art performance (**93.7% Test Accuracy**) by averaging softmax probabilities across multiple fine-tuned ResNet models.

## 📊 Dataset

* **Total Images:** 17,034 RGB images (150x150 pixels).
* **Split:** 11,227 Training / 2,807 Validation / 3,000 Test.
* **Classes:** 6 balanced categories.
* **Preprocessing:** Resized to 224x224, converted to Tensors, and normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

## 🧠 Architectures & Approaches

### 1. Supervised Classification
* **Baseline CNN:** A custom 4-block Convolutional Neural Network (~8.7M parameters) with Batch Normalization and MaxPooling.
* **Transfer Learning (`models/transfer_models.py`):** * Integrated PyTorch `torchvision.models`.
    * Implemented custom layer unfreezing logic for fine-tuning specific blocks (e.g., `layer4` in ResNets) with separate learning rates for feature extractors and classification heads.

### 2. Unsupervised Representation Learning (`models/autoencoder.py`)
* **Convolutional Autoencoder (CAE):** Trained to compress 224x224 images into a 128-dimensional latent space. Visualized class clustering using **t-SNE**.
* **Denoising Autoencoder (DAE):** Trained to reconstruct images corrupted by Gaussian noise (noise factor = 0.3). Reconstruction quality was evaluated using Peak Signal-to-Noise Ratio (PSNR).

## 📈 Performance & Results

The models were evaluated based on their Validation and Test Accuracy, as well as their ability to mitigate overfitting (Train Acc - Val Acc).

| Model / Experiment | Mode | Regularization / Augmentation | Test Accuracy | Overfitting Gap |
| :--- | :--- | :--- | :---: | :---: |
| Baseline Custom CNN | From Scratch | Baseline | 86.60% | ~ 5.3% |
| EfficientNet-B0 | Frozen | Baseline | 89.40% | N/A |
| ResNet50 | Frozen | Baseline | 91.20% | N/A |
| ResNet18 | Finetuned | Baseline | 92.17% | + 0.1% |
| ResNet18 | Finetuned | Advanced Augmentation | 92.87% | - 1.0% |
| ResNet18 | Finetuned | Advanced Aug + Label Smoothing | 93.03% | + 0.9% |
| ResNet18 | Finetuned | All (Aug + Smooth + Dropout + L2) | 93.30% | + 0.2% |
| **ResNet Ensemble (x3)** | **Softmax Avg** | **Mixed Strategies** | **93.70%** | **N/A** |

*Note: The final ensemble averaged the predictions of a baseline-augmented ResNet18, an advanced-augmented ResNet18, and a heavily regularized ResNet50.*

## 🛠️ Tech Stack & Libraries

* **Core:** Python 3, PyTorch (`torch`, `torchvision`, `nn.Module`)
* **Data Processing:** NumPy, Pandas, PIL
* **Machine Learning Metrics:** Scikit-Learn (t-SNE, Confusion Matrix, Classification Report)
* **Visualization:** Matplotlib, Seaborn
* **Hardware:** Trained utilizing CUDA GPU acceleration.

## 📂 Project Structure

```text
├── augmentation/
│   ├── advanced_aug.py         # Custom transforms, Data Augmentation, Label Smoothing loss, Dropout wrapper
├── models/
│   ├── autoencoder.py          # ConvAutoencoder and DenoisingAutoencoder classes
│   ├── custom_cnn.py           # Baseline CNN architecture
│   ├── transfer_models.py      # Transfer learning wrapper with selective layer unfreezing
├── Jupiter/                    # Experimental Jupyter Notebooks
│   ├── dataset_check.ipynb     # EDA, image size distribution, and class balance checks
│   ├── baseline_cnn.ipynb      # Training from scratch
│   ├── transfer_learning.ipynb # Frozen vs Finetuning experiments
│   ├── augmentation.ipynb      # Regularization strategy comparisons
│   ├── autoencoder.ipynb       # Unsupervised learning, PSNR, and t-SNE visualization
│   └── final_comparison.ipynb  # Softmax ensembling and final evaluation
├── results/                    # Exported plots, learning curves, and CSV reports
└── README.md                   # Project documentation
