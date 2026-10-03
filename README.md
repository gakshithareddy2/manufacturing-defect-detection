# Manufacturing Defect Detection

AI-based steel surface defect classification using a fine-tuned ResNet50 deep learning model.

## 🚀 Live Demo

[Open the Manufacturing Defect Detection Dashboard](https://manufacturing-defect-detection-zr5asxphznfacqchsayq6p.streamlit.app/)

## 📌 Project Overview

This project uses Deep Learning and Transfer Learning to automatically classify defects present on steel surface images.

The system uses a fine-tuned ResNet50 model trained on the NEU Surface Defect Dataset.

## 🎯 Objective

To develop an automated manufacturing defect detection system that can classify steel surface defects from uploaded images.

## 🧠 Model

**Fine-tuned ResNet50**

The model classifies images into six defect categories:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## 📊 Model Performance

| Model | Validation Accuracy |
|---|---:|
| Baseline CNN | 42.50% |
| Regularized CNN | 32.22% |
| ResNet50 Transfer Learning | 98.89% |
| Fine-tuned ResNet50 | 99.17% |

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- ResNet50
- NumPy
- Streamlit
- Hugging Face Hub
- Google Colab

## 🔄 How It Works

Steel Surface Image
        ↓
Image Upload
        ↓
Preprocessing
        ↓
Fine-tuned ResNet50
        ↓
Defect Classification
        ↓
Predicted Defect + Confidence
        ↓
Class Probabilities
