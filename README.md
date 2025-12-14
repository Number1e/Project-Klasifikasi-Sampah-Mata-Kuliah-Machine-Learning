# Project-Klasifikasi-Sampah-Mata-Kuliah-Machine-Learning
# ♻️ Trash Classification: Comparative Study (ViT vs CNN vs SVM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project focuses on the classification of waste materials to support automated recycling processes. The core objective is not just to build a classifier, but to conduct a **comparative analysis** between Deep Learning paradigms and Traditional Machine Learning:

1.  **CNN from Scratch:** Establishing a baseline with a lightweight custom architecture.
2.  **Transfer Learning:** Leveraging pre-trained knowledge for high efficiency.
3.  **Vision Transformer (ViT):** Exploring the efficacy of Self-Attention mechanisms on waste datasets.
4.  **Support Vector Machine (SVM):** Comparing modern Deep Learning with classical Machine Learning methods.

This repository contains the full pipeline: from data loading, model training, to a **Streamlit GUI** for easy inference.

## 📂 Dataset
The dataset consists of **2500** images divided into **3** classes:
* **Kaca (Glass)**
* **Kardus (Cardboard)**
* **Plastik (Plastic)**

> **Note:** Due to storage limitations, the raw dataset is hosted externally on Google Drive.

### How to Setup Data
1.  **Download Manual:**
    Download the dataset from [Google Drive Link](https://drive.google.com/drive/u/0/folders/1q5WNdPgmqbhq93u7WSyaScZyE3LUuYSg).
2.  **Extract:**
    Extract the contents into the `data/raw/` directory so the structure becomes:
    ```
    data/raw/
    ├── Kaca/
    ├── Kardus/
    └── Plastik/
    ```

## 🏗️ Methodologies

### 1. Custom CNN (Baseline)
* **Architecture:** Custom 4-Block Convolutional Neural Network.
* **Goal:** To test performance without any external knowledge (weights).
* **Characteristics:** Optimized for local feature extraction with significantly lower parameter count than standard VGG/ResNet.

### 2. Transfer Learning (CNN)
* **Backbone:** **MobileNetV2** (Pre-trained on ImageNet).
* **Strategy:** Freezing the feature extractor layers and fine-tuning the classification head.
* **Goal:** To maximize accuracy with limited training data.

### 3. Vision Transformer (ViT)
* **Model:** A custom-built, lightweight Vision Transformer (ViT) implemented from scratch.
* **Configuration:** Input size: 128x128 (resized for efficiency), Patch size: 16x16.
* **Strategy:** Patch-based Learning with Self-Attention mechanisms.
* **Goal:** To analyze the efficacy of the Attention mechanism on waste recognition tasks under constrained computational resources (non-pretrained).

### 4. Support Vector Machine (SVM)
* **Feature Extraction:** Manual extraction using Color Histograms (HSV), Texture (GLCM), and Shape (Hu Moments).
* **Kernel:** RBF (Radial Basis Function).
* **Goal:** To provide a benchmark using classical computer vision techniques.

## 📊 Experimental Results

We trained each model for **20** epochs. Below is the performance comparison on the Test Set:

| Model Architecture | Accuracy | F1-Score |
| :--- | :---: | :---: |
| **Transfer Learning (MobileNetV2)** | **89.0%** | **0.89** |
| Custom CNN (Scratch) | 64.0% | 0.64 |
| SVM (Traditional ML) | 68.0% | 0.68 |
| Vision Transformer (ViT Scratch) | 59% | 0.59 |

### 💡 Key Insights
* **Transfer Learning Dominance (89%):** MobileNetV2 achieved superior performance because it leverages features learned from millions of images (ImageNet). For a small dataset (2500 images), pre-trained weights prevent overfitting and converge much faster.
* **CNN vs ViT (64% vs 59%):** The Custom CNN significantly outperformed the Custom ViT. This supports the theory that **ViTs lack inductive bias** (translation invariance and locality) inherent in CNNs. ViTs typically require massive datasets (JFT-300M, ImageNet-21k) to learn spatial relationships effectively. On a small dataset trained from scratch, ViT fails to generalize well.
* **SVM Performance (68%):** SVM performed respectably, beating ViT. This shows that for small datasets, hand-crafted features (Color/Texture) can sometimes be more effective than a deep learning model that hasn't converged properly (like the ViT scratch).

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* GPU (Optional, but recommended for training)

### 1. Clone & Install
```bash
git clone [https://github.com/Number1e/Project-Klasifikasi-Sampah-Mata-Kuliah-Machine-Learning.git](https://github.com/Number1e/Project-Klasifikasi-Sampah-Mata-Kuliah-Machine-Learning.git)
cd Project-Klasifikasi-Sampah-Mata-Kuliah-Machine-Learning
pip install -r requirements.txt
```
### 2. Run the App (GUI)
```bash
streamlit run app.py
```
> **Note:** for Google Colab users Since Colab runs on the cloud, you cannot access localhost directly. You must use Ngrok tunneling as described in the notebook to view the dashboard.


## 👥 Authors

**1. [Moh. Zahidi]**
* **NIM:** [202210370311419]
* **University:** [Universitas Muhammadiyah Malang]

**2. [Haris Rifky Juliantoro]**
* **NIM:** [202210370311421]
* **University:** [Universitas Muhammadiyah Malang]

**3. [Syafrizal Rabbanie]**
* **NIM:** [202210370311453]
* **University:** [Universitas Muhammadiyah Malang]
