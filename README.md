# Intelligent Intrusion Detection System (IDS) for Internet of Things (IoT) Networks Using Machine Learning
Leveraging machine learning, this project focuses on identifying network intrusions and malicious behavior within IoT environments. Utilizing the CIC-IoT2023 dataset, it incorporates data preprocessing, feature engineering, class balancing, and model training to achieve accurate and real-time prediction of threats in IoT deployments.


---

## Table of Contents
- ⁠[Project Structure](#project-structure)
- ⁠[Dataset](#dataset)
-  ⁠[Overview](#overview)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Features](#features)
- ⁠[Results](#results)
-  ⁠[Team](#team)

---
## Project Structure
- notebook/ ⁠: Jupyter Notebook for data preprocessing, feature engineering, and model training⁠
- datasets/ ⁠: CSV files from CIC-IoT 2023 dataset
- ⁠⁠requirements.txt ⁠: Python libraries required
- ⁠⁠ README.md ⁠: Project documentation
---
## Dataset

We used the [CIC-IoT2023 Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html) which contains both benign and malicious IoT traffic. The dataset includes attacks like DDoS, Injection, Reconnaissance, etc.

---
## Overview
### Phase 1: Data Preprocessing & Feature Engineering

- ⁠Cleaned missing values (handled NaNs and infinite values)
- ⁠Feature normalization using *Z-score (StandardScaler)*
-  ⁠Balanced dataset using *SMOTE* to fix class imbalance
-   ⁠Reduced dimensionality using *PCA* (Principal Component Analysis)

#### Tools Used:
- pandas ⁠, ⁠ numpy ⁠
- scikit-learn ⁠
-  imbalanced-learn ⁠ (SMOTE)
- ⁠⁠ matplotlib ⁠, ⁠ seaborn ⁠ (for optional EDA/visualization)
-  ⁠⁠ sklearn.decomposition ⁠ (PCA)
-   ⁠⁠ PyCaret ⁠ (optional end-to-end pipeline support)

### Phase 2: ML Modeling & Evaluation

#### Supervised Models Trained:
- ⁠Random Forest
- ⁠Support Vector Machine (SVM)
- ⁠XGBoost
- ⁠Multilayer Perceptron (MLP - Neural Network)
- ⁠Convolutional Neural Network (CNN)

#### Unsupervised Models Trained:
- ⁠Autoencoder (deep learning)
- ⁠Isolation Forest (anomaly detection)

#### Tools Used:
- scikit-learn ⁠
- XGBoost ⁠, ⁠ LightGBM ⁠
- TensorFlow ⁠ / ⁠ Keras ⁠

---
## Installation
1.⁠ ⁠Clone the repository:
⁠  bash
git clone https://github.com/CelineHarakee/IoT-Cyber-Attack-Detection-Using-ML.git
cd IoT-Cyber-Attack-Detection-Using-ML
 ⁠
2.⁠ ⁠Install dependencies:
⁠  bash
pip install -r requirements.txt
 ⁠
---
## How to Run
Run the notebook step by step: 
⁠  bash
notebooks/ML_Final_Project_Code.ipynb 
 ⁠
--- 
## Features
This project performs intelligent intrusion detection on IoT network traffic using machine learning. Key features include:
- ⁠*Data Cleaning & Preprocessing*
  - Handled missing values (cleaned columns with NaNs).
  - Applied Z-score normalization using StandardScaler.
- ⁠*Feature Engineering*
  - Encoded attack labels: Benign, DDoS, Injection, Recon.
  - Balanced data with SMOTE to address class imbalance.
  - Reduced feature dimensionality using PCA.
- ⁠*Model Training*
  - Trained and compared four supervised ML models: Random Forest, Support Vector Machine (SVM), XGBoost, Multilayer Perceptron (MLP).
  - Implemented optional unsupervised anomaly detection: Isolation Forest, Autoencoder (Keras).
- ⁠*Evaluation Metrics*
  - Used Precision, Recall, F1-score, and Accuracy for model comparison.
  - Visualized performance with Confusion Matrices.
---
## Results
Each model was evaluated on how well it detects attack types using classification metrics:
| Model                | Precision | Recall | F1-score | Accuracy |
|----------------------|-----------|--------|----------|----------|
| *Random Forest*     | 0.98      | 0.98   | 0.98     | 98.0%    |
| *SVM*               | 0.97      | 0.96   | 0.96     | 96.5%    |
| *XGBoost*           | 0.98      | 0.97   | 0.97     | 97.0%    |
| *MLP Neural Network*| 0.97      | 0.97   | 0.97     | 97.2%    |

#### Insights:
- ⁠*Random Forest* performed consistently across both frequent and rare attack types.
- *SVM* was fast and effective but struggled slightly with rare class detection.
- ⁠*XGBoost* provided great generalization and fast training.
- ⁠*MLP* learned complex patterns well and matched XGBoost’s performance.
---
## Team
•⁠  ⁠[Celine Al Harake](https://github.com/CelineHarakee)
•⁠  ⁠[Layal Canoe](https://github.com/layalcanoe)
•⁠  ⁠[Dana Al Rijjal](https://github.com/daaalrijjal)
•⁠  ⁠[Jouri Al Daghma](https://github.com/Jourialdagh)
