# Dermatology Classification

## Overview
This project tackles the problem of multi-class classification of six types of erythemato-squamous skin diseases:
1. Psoriasis
2. Seborrheic Dermatitis
3. Lichen Planus
4. Pityriasis Rosea
5. Chronic Dermatitis
6. Pityriasis Rubra Pilaris

These conditions appear very similar clinically, as they all present with erythema and scaling. Thus, distinguishing them based on symptoms alone is difficult. This project leverages a combination of clinical and histopathological features to train machine learning models that can accurately classify the diseases.

## Dataset
- **Source:** [Kaggle — Dermatology Dataset](https://www.kaggle.com/datasets/olcaybolat1/dermatology-dataset-classification/data) (originally from the UCI Machine Learning Repository).
- **Size:** 366 instances (patients).
- **Features:** 34 features (12 clinical and 22 histopathological).
- **Target:** A class label from 1 to 6 corresponding to the disease type.

## Project Structure
- `main.ipynb`: The main Jupyter Notebook containing the full pipeline: data downloading, preprocessing, exploratory data analysis (EDA), model training, evaluation, and experimental conclusions.

## Methodology

### 1. Data Preprocessing
- **Missing Value Imputation:** Missing values in the `age` column are handled using a **KNN-based Imputer** (`k=5`), providing a more robust approach compared to simple median imputation.
- **Scaling:** Data is standardized using `StandardScaler` to ensure features have zero mean and unit variance.
- **Data Splitting:** Data is split into 80% training and 20% test sets, using stratification to handle mild class imbalances (such as the rare Pityriasis Rubra Pilaris class).

### 2. Exploratory Data Analysis (EDA)
- Analyzed class distribution using bar charts.
- Generated a feature correlation matrix to evaluate relationships among the clinical and histopathological variables.

### 3. Model Training & Hyperparameter Tuning
Several machine learning classifiers were evaluated. Hyperparameters were tuned systematically using `GridSearchCV`:
- **SVM (Linear Kernel):** Tuned regularization strength `C`.
- **SVM (RBF Kernel):** Tuned `C` and the kernel coefficient `gamma`.
- **Random Forest:** Tuned the number of estimators, max depth, and minimum samples required for splitting.

### 4. Dimensionality Reduction (PCA)
Principal Component Analysis (PCA) was integrated into the pipeline to study the impact of dimensionality reduction on classification performance. Models were retrained on PCA-reduced data and compared with their full-feature counterparts.

### 5. Evaluation
Models were rigorously assessed using:
- Cross-validation accuracy during Grid Search.
- Test accuracy.
- Detailed Classification Reports (Precision, Recall, F1-Score).
- Confusion Matrices to visualize model misclassifications.

