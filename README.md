# Feeding Mode Classification and Modeling Based on Machine Learning

This project focuses on the classification of feeding modes using multiple machine learning models, including **Artificial Neural Networks (ANN)**, **Random Forest (RF)**, and **Support Vector Machines (SVM)**. The dataset consists of structured features extracted from an Excel file containing different feeding patterns and their corresponding labels.

## 📂 Project Overview

The main goal of this project is to:
- Build and evaluate classification models to distinguish different feeding methods.
- Compare the performance of ANN, RF, and SVM through hyperparameter tuning.
- Visualize results using confusion matrices and 3D plots to analyze model performance under different parameter settings.

All results are generated and saved as `.pdf` figures for further comparison and analysis.

## ⚙️ Installation and Environment

This project requires the following dependencies:

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `torch`
- `matplotlib`
- `seaborn`
- `openpyxl`

To install the required packages: ```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn openpyxl


## 🧠 Main Functionalities
The codebase contains three main modeling approaches:

1.Artificial Neural Network (ANN)

Built with PyTorch

Performs hyperparameter tuning with ParameterGrid

Visualizes accuracy scores in 3D

Outputs best model’s confusion matrix

2.Random Forest Classifier

Uses GridSearchCV for hyperparameter optimization

Evaluates best model performance

Visualizes confusion matrix and parameter grid in 3D

3.Support Vector Machine (SVM)

Uses GridSearchCV to search over C, gamma, and kernel

Compares best and poor performing parameter combinations

Visualizes confusion matrices and hyperparameter scores in 3D

Each model’s output includes:

Classification report

Confusion matrix (saved as PDF)

3D parameter-performance plots

If you have any questions or suggestions, feel free to open an issue or contact the author. Thank you for your interest!
