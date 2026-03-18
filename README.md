# Bank Customer Churn Prediction

> End-to-end machine learning project predicting customer churn for a retail bank, covering the full data science lifecycle from raw data to business-ready insights.

## Project Overview

Customer churn — the loss of clients to competitors — costs retail banks an estimated **€800 per customer per year** in lost revenue. This project builds a classification model to identify at-risk customers before they leave, and translates model outputs into actionable, revenue-optimised retention strategies.

**Dataset:** 10,000 customers | 14 features | Binary target (Exited)

## Repository Structure

```
bank-customer-churn-prediction/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis & data validation
│   ├── 02_features.ipynb         # Feature engineering & train/test split
│   ├── 03_modelling.ipynb        # Model training, tuning & evaluation
│   ├── 04_explainability.ipynb   # SHAP values & model interpretability
│   └── 05_business.ipynb         # Threshold optimisation & revenue impact
├── data/
│   ├── raw/                      # Original, immutable data
│   ├── processed/                # Cleaned & transformed data
│   └── external/                 # Third-party data sources
├── models/                       # Serialised model files (.pkl / .joblib)
├── reports/
│   └── figures/                  # Generated charts and visualisations
├── src/                          # Reusable helper modules
├── .gitignore
└── README.md
```

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | EDA | Distribution analysis, missing value checks, churn rate breakdown by segment |
| 02 | Feature Engineering | Business-driven features, one-hot encoding, stratified train/test split |
| 03 | Modelling | Logistic Regression, Random Forest, XGBoost — cross-validated with ROC-AUC |
| 04 | Explainability | Global & local SHAP explanations, feature importance ranking |
| 05 | Business Value | Cost-sensitive threshold selection, revenue optimisation analysis |

## Key Results

- **Best model:** XGBoost — ROC-AUC 0.87
- **Optimal threshold:** Tuned for business cost function (false negative = €800, false positive = €30)
- **Revenue impact:** Model-driven retention strategy recovers an estimated **€X per 1,000 customers** vs. no intervention

## Tech Stack

```
Python 3.10+  |  pandas  |  scikit-learn  |  XGBoost  |  SHAP  |  matplotlib  |  seaborn
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/TheEmeraldAgent/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/
```

## Author

**Bora** — Data Analyst  
[GitHub](https://github.com/TheEmeraldAgent)
