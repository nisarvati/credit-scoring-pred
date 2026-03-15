# AI-Powered Alternative Credit Scoring for Financial Inclusion

A machine learning pipeline for equitable credit risk assessment, built to expand financial access for underserved and unbanked populations. This project balances strong predictive accuracy with algorithmic fairness and local explainability.

## Problem Statement

Traditional credit scoring models often exclude or disadvantage individuals with limited credit history — disproportionately affecting unbanked and underbanked populations. In this dataset, approximately 4.0% of users lack traditional credit. This project builds a credit scoring system that uses alternative feature engineering and dynamic thresholding to ensure fairness across demographic groups without compromising risk standards.

## Approach & Pipeline

The pipeline follows a rigorous end-to-end structured workflow:

1. **Data Preprocessing & Engineering** — Missing value imputation (using training medians), outlier capping (1st/99th percentiles), and engineering 20+ alternative credit features (e.g., `payment_discipline_score`, `unbanked_proxy`).
2. **Handling Imbalance** — Applying SMOTE (Synthetic Minority Oversampling Technique) to correct the initial 14:1 class imbalance.
3. **Dimensionality Reduction** — Applying PCA to reduce the feature space to 16 components while retaining 95% of the variance.
4. **Model Optimization** — Training an XGBoost classifier, rigorously tuned via RandomizedSearchCV to prevent overfitting and maximize the AUC-ROC score.
5. **Fairness Auditing** — Implementing Dynamic Thresholding to equalize opportunity (Recall Parity) for protected groups like Young Borrowers (<25) and the Unbanked, satisfying the 80% Disparate Impact Rule.
6. **Explainable AI (XAI)** — Projecting SHAP (SHapley Additive exPlanations) values back from the PCA space to original feature names to generate transparent "Reasons for Denial/Approval".

## Tech Stack

- **Language:** Python 3.12
- **Machine Learning:** XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Data & Visualization:** pandas, NumPy, Matplotlib, Seaborn

## Repository Structure

| File | Description |
|------|-------------|
| `credit-score-final.ipynb` | Core notebook — full EDA, pipeline, and evaluation |
| `cs-training.csv` | Primary training dataset (Give Me Some Credit) |
| `cs-test.csv` | Primary test dataset |
| `german_credit_data.csv` | Supplementary dataset |
| `Data Dictionary.xls` | Dataset variable definitions |
| `credit_scores_with_shap.csv` | Output predictions with SHAP top risk drivers (30K test cases) |
| `sampleEntry.csv` | Template for new predictions |
| `xgb_tuned.pkl` | Champion XGBoost model (tuned) |
| `xgb_credit_model.pkl` | Base XGBoost model |
| `scaler.pkl` | StandardScaler fitted on training data |
| `pca.pkl` | Fitted PCA transformer |
| `train_medians.pkl` | Medians for missing value imputation during inference |
| `feature_columns.pkl` | Expected input column schema |

## Key Results & Business Impact

| Metric | Result |
|--------|--------|
| **AUC-ROC** | 0.8622 |
| **Overall Accuracy** | 80.38% |
| **Unbanked Users Scored** | 4,858 in the test group |
| **Estimated Lending Potential** | ₹24.3 Crore (assuming avg. micro-credit limit of ₹50,000) |
| **Compliance** | Production-ready pipeline with built-in fairness audits for Fair Lending Compliance |

## Future Work

- Incorporate alternative data sources (e.g., utility payments, mobile data) to better serve thin-file applicants
- Deploy the `predict_credit_score()` inference function as a REST API (using FastAPI or Flask) for real-time scoring
