# Credit Card Fraud Detection (ML + Imbalanced Data)

Detect fraudulent credit card transactions using machine learning, with a focus on handling **extreme class imbalance**.  
This project walks through **EDA → preprocessing → SMOTE oversampling → model training → evaluation**, and includes optional experiments (tree models, anomaly detection, deep learning) plus a simple deployment example.

---

## What this project does

- Loads and explores the dataset (`creditcard.csv`)
- Checks **fraud vs non-fraud distribution** (highly imbalanced)
- Checks and removes **missing values**
- Flags possible **outliers** in `Amount` using IQR
- Visualizes:
  - Transaction amount distribution
  - Correlation heatmap
- Balances the dataset using **SMOTE**
- Trains a baseline **Logistic Regression** model
- Evaluates with **Precision / Recall / F1**
- (Optional sections) Additional modeling ideas:
  - Random Forest + Gradient Boosting + simple ensemble vote
  - Isolation Forest (unsupervised anomaly detection)
  - Deep learning experiments (LSTM / Autoencoder concepts)
- (Optional) Saves a trained model with `pickle` + example **Flask API** endpoint

---

## Dataset

This notebook expects a CSV named:

- `creditcard.csv`

> Note: Many repos don’t include this CSV because it can be large or has download restrictions.  
> If you don’t have it locally, download the “Credit Card Fraud Detection” dataset (commonly hosted on Kaggle) and place it in the project root.

---

## Results (Baseline)

After balancing with **SMOTE** and training a **Logistic Regression** classifier, the notebook reports:

- **Precision:** `0.9904`
- **Recall:** `0.9672`
- **F1 Score:** `0.9787`

> Since fraud detection is usually more sensitive to *missed fraud*, recall is especially important.

---

## Project Structure

```text
.
├── CreditCardFraudDetection.ipynb
├── creditcard.csv                # (you need to add/download this)
└── README.md
