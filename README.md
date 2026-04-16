# CODSOFT Task 5 - Credit Card Fraud Detection

## Objective
Detect fraudulent credit card transactions in a highly imbalanced dataset using machine learning.

## Dataset
This project uses the Credit Card Fraud Detection dataset from Kaggle.

**Dataset Link**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Details**: 284,807 transactions, 492 frauds (0.172%). Features: Time, V1-V28 (PCA transformed), Amount. Target: Class (0=Normal, 1=Fraud).

**Instructions**: 
1. Download `creditcard.csv` from the Kaggle link above
2. Place it in the same folder as `Credit_Card_Fraud_Detection.ipynb`
3. Run the notebook

**Note**: The dataset is not included in this repo due to GitHub size limits and Kaggle's Terms of Service.

## Approach
1. Handled class imbalance using `class_weight='balanced'`
2. Scaled `Time` and `Amount` features. V1-V28 already scaled by PCA.
3. Trained Logistic Regression and Random Forest Classifier
4. Evaluated using ROC-AUC, Precision, Recall - not Accuracy

## Final Results
- **Best Model**: Logistic Regression
- **ROC-AUC Score**: 0.972
- **Random Forest AUC**: 0.953

## Key Insights
The dataset is highly imbalanced, so AUC-ROC and Recall are the right metrics. Logistic Regression slightly outperformed Random Forest. Top predictive features are V14, V10, V12, V4, V17 - anonymized PCA components that strongly separate fraud
