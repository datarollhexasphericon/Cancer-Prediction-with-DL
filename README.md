# Breast Cancer Prediction Using Deep Learning

## Project Motivation and Description

This project aims to leverage deep learning techniques to develop an accurate and robust model for breast cancer classification. By analyzing various tumor measurements, the model predicts whether a tumor is benign or malignant, aiding early detection and diagnosis, which are crucial for effective treatment and improving patient outcomes. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering (PCA), model training, hyperparameter tuning, and evaluation.

## Dataset

The dataset consists of various features derived from medical images of breast tumors. This is a synthetic dataset that can be found at https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/code.

## Key Components

### Data Preparation and Exploration
- Loading and preprocessing the dataset.
- Performing exploratory data analysis (EDA) to understand data distribution and feature relationships.
- Scaling features to ensure equal contribution to the model's performance.

### Model Definition
- Designing a fully-connected neural network using PyTorch.
- The network consists of three fully connected layers with ReLU activations and a sigmoid activation function at the output layer.

### Training and Hyperparameter Tuning
- Training the model with various combinations of learning rates, batch sizes, and number of epochs.
- Implementing early stopping to prevent overfitting.
- Identifying the optimal model configuration.

### Model Evaluation
- Evaluating the model's performance using metrics such as accuracy, F1 score, and ROC AUC.
- Visualizing the confusion matrix and ROC curve.

## Conclusion
This project demonstrates the use of deep learning for breast cancer prediction, achieving high accuracy and robustness. The developed model shows promise for aiding medical professionals in early detection and diagnosis of breast cancer.






