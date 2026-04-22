# Vehicle Insurance Fraud Detection

A final year undergraduate machine learning project that builds, evaluates and interprets a pipeline for detecting fraudulent automobile insurance claims. The project addresses key real-world challenges including severe class imbalance (~6% fraud rate), cost-sensitive decision making, and operational threshold selection.

## Notebooks

- `01_eda_auto_insurance_fraud.ipynb` — Exploratory data analysis. Covers class imbalance, fraud rate by key features (fault, accident area, policy type, vehicle price, evidence presence), temporal patterns, age group analysis, and validation of engineered features.

- `02_data_prep_auto_insurance_fraud.ipynb` — Data cleaning and feature engineering. Includes midpoint conversions for banded ordinal columns, five domain-driven engineered features (ClaimDelay, PriceToDeductibleRatio, IsWeekendAccident, LowPriceHighInsured, NoEvidence), and two interaction features for target encoding.

- `03_modelling_auto_insurance_fraud.ipynb` — Full ML pipeline including:
  - Baseline comparison: Logistic Regression, XGBoost, LightGBM, CatBoost
  - Imbalance handling: class weights, RandomOverSampler, SMOTENC, ADASYN
  - Hyperparameter tuning: RandomizedSearchCV (30 iterations, PR-AUC scoring)
  - Threshold optimisation using out-of-fold probabilities
  - Business cost evaluation at multiple operating points
  - Fairness check by claimant sex
  - SHAP interpretability (global feature importance + individual waterfall plots)
  - Temporal validation (train on months 1–18, test on months 19–24)

## Results Summary

| Metric | Value |
|--------|-------|
| Final model | LightGBM (tuned, class weights) |
| AUC-ROC | 0.8386 |
| PR-AUC | 0.2255 |
| F1 | 0.2911 |
| Fraud caught (Recall) | 54.1% (100/185) |
| Selected threshold | 0.61 |

## Dataset

- Source: [Vehicle Claim Fraud Detection — Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- 15,420 auto insurance claims (1994–1996)
- Fraud rate: ~6% (923 fraudulent claims)

## Framework

CRISP-DM — implemented in Python on Google Colab.

## Author

Yahya Habib
