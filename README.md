# Medical-Image-Classification-Brain-Tumor
# Medical Image Classification using Deep Learning

## Overview
A deep learning model for classifying medical images (X-rays) into different disease categories using CNN architecture and transfer learning with ResNet50.

## Disease Categories
- Normal
- Pneumonia
- COVID-19
- Tuberculosis

## Key Features
- Data preprocessing pipeline for medical images
- Transfer learning with ResNet50
- Grad-CAM visualization for model interpretability
- Performance metrics and ROC curves
- Web interface for image upload and prediction

## Tech Stack
- Python 3.8+
- PyTorch
- OpenCV
- scikit-learn
- Streamlit
- Matplotlib/Seaborn

## Project Structure

'''

Medical-Image-Classification-AI/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── processed/
│
├── models/
│   ├── saved_models/
│   └── checkpoints/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualization.py
│   └── utils.py
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Development.ipynb
│
├── webapp/
│   ├── app.py
│   └── static/
│
├── tests/
│   └── test_model.py
│
├── requirements.txt
├── config.yaml
└── README.md

'''
