# vehicle-insurance-fraud-detection

An industry-relevant machine learning project that flags potentially fraudulent automobile insurance claims. It focuses on key real-world challenges such as class imbalance and evaluation trade-offs (investigation workload vs missed-fraud cost).

## Contents
- `notebook/EDA_missing_values.ipynb` — EDA, missing-value checks, data preparation, and feature encoding (binary, ordinal mapping, one-hot encoding).
- `notebook/XGBoost_CatBoost.ipynb` — Model training and evaluation:
  - Baseline (no SMOTE)
  - SMOTE
  - SMOTE + cost-sensitive learning (sample weights)
  - Threshold analysis (top-k% alerts), cost scenario analysis, and standard evaluation metrics.

## Dataset
- Link: **https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection**
