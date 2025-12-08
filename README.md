# Project-Klasifikasi-Sampah-Mata-Kuliah-Machine-Learning
# ♻️ Trash Classification: Comparative Study (ViT vs CNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project focuses on the classification of waste materials to support automated recycling processes. The core objective is not just to build a classifier, but to conduct a **comparative analysis** between three distinct Deep Learning paradigms:

1.  **CNN from Scratch:** Establishing a baseline with a lightweight custom architecture.
2.  **Transfer Learning:** Leveraging pre-trained knowledge for high efficiency.
3.  **Vision Transformer (ViT):** Exploring the efficacy of Self-Attention mechanisms on waste datasets.

This repository contains the full pipeline: from data loading (Google Drive integration) to model training and evaluation metrics.

## 📂 Dataset
The dataset consists of **[2500]** images divided into **[3]** classes (e.g., *Plastic, Paper, Metal, Glass, etc.*).

> **Note:** Due to storage limitations, the raw dataset is hosted externally on Google Drive.

### How to Setup Data
We provide a script to automatically download and arrange the data:

1.  Run the setup script:
    ```bash
    python src/download_data.py
    ```
2.  Alternatively, download manually from **[https://drive.google.com/drive/u/0/folders/16TLvMp39vcos9LCal64AFJW0DOLDfc07](https://drive.google.com/drive/u/0/folders/1q5WNdPgmqbhq93u7WSyaScZyE3LUuYSg)** and extract it to the `data/raw/` directory.

## 🏗️ Methodologies

### 1. Custom CNN (Baseline)
* **Architecture:**Custom 4-Block Convolutional Neural Network.
* **Goal:** To test performance without any external knowledge (weights).
* **Characteristics:** Optimized for local feature extraction with significantly lower parameter count than standard VGG/ResNet.

### 2. Transfer Learning (CNN)
* **Backbone:** **MobileNetV2** (Pre-trained on ImageNet).
* **Strategy:** Freezing the feature extractor layers and fine-tuning the classification head.
* **Goal:** To maximize accuracy with limited training data.

### 3. Vision Transformer (ViT)
* **Model:** A custom-built, lightweight Vision Transformer (ViT) implemented from scratch.
* **Configuration:** Input size: 128x128 (resized for efficiency).
* **Strategy:** Patch-based Learning. Images are split into 16x16 patches and processed via Self-Attention mechanisms to capture global context.
* **Goal:** To analyze the efficacy of the Attention mechanism on waste recognition tasks under constrained computational resources (non-pretrained).

## 📊 Experimental Results

We trained each model for **20** epochs. Below is the performance comparison on the Test Set:

| Model Architecture | Accuracy | F1-Score | 
| :--- | :---: | :---: | 
| **Custom CNN** | 74% | 0,74 |
| **Transfer Learning** | **96%** | **o,96** |
| **Vision Transformer** | 60,5% | o,6 |
| **SVM** | 68% | o,68 |

### 💡 Key Insights
* **Transfer Learning Dominance:** [Jelaskan hasilmu, misal: Transfer learning achieved the highest accuracy due to robust feature extraction learned from ImageNet.]
* **ViT vs CNN:** [Jelaskan hasilmu, misal: ViT struggled slightly/performed well. Note that ViTs generally lack inductive bias (translation invariance) and typically require massive datasets to outperform CNNs significantly.]
* **Efficiency:** [Jelaskan mana yang paling cepat/ringan].

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* CUDA capable GPU (Recommended)

### 1. Clone & Install
```bash
git clone https://github.com/Number1e/Project-Klasifikasi-Sampah-Mata-Kuliah-Machine-Learning.git.git
pip install -r requirements.txt
# Train Custom CNN
python src/train.py --model custom_cnn --epochs 20 --batch_size 32

# Train Transfer Learning Model
python src/train.py --model transfer_learning --epochs 20 --batch_size 32

# Train Vision Transformer
python src/train.py --model vit --epochs 20 --batch_size 16
python src/evaluate.py --model_path models/best_model.pth
