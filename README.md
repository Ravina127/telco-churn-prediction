# Telco Customer Churn Prediction

Predicts which telecom customers are likely to leave (churn), explains the main drivers of churn, and estimates the business value of targeting high-risk customers. Includes a Streamlit app that scores a customer from their details.

**Live app:** _add your Streamlit link here after deploying_

## Problem
Winning a new customer costs far more than keeping an existing one. The goal is to find at-risk customers early, so the company can offer them the right help before they leave.

## Data
Telco Customer Churn dataset: 7,043 customers and 21 columns (demographics, account details, services, and the target `Churn`). About 26.5% of customers churned, so the data is imbalanced.

## Approach
1. Data cleaning (fixed `TotalCharges`, encoded the target, dropped `customerID`)
2. Exploratory data analysis (EDA)
3. Feature engineering (`NumAddOns`, `AvgMonthlySpend`)
4. Model comparison with 5-fold cross-validation (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
5. Hyperparameter tuning, then evaluation once on an untouched test set
6. Threshold selection, permutation importance and business impact

Preprocessing (scaling and one-hot encoding) sits inside a scikit-learn pipeline that is fitted on the training data only, so no information leaks from the test set.

## Key results
| Item | Result |
|---|---|
| Overall churn rate | 26.5% |
| Best model | Gradient Boosting (tuned) |
| Test ROC-AUC | 0.845 |
| Decision threshold 0.40 | Catches 64% of churners with 60% precision |
| Top 20% riskiest customers | Contain about half of all churners (2.5x better than random) |

**Top churn drivers:** contract type, tenure, internet service, tech support, online security.

**Highest-risk group:** month-to-month contract with fibre optic internet (55% churn).

## Recommendations
- Move month-to-month customers to 1 or 2-year contracts
- Add onboarding support in the first 12 months, when churn is highest (about 47%)
- Review fibre-optic quality and pricing
- Encourage automatic payments (electronic-check users churn at 45%)
- Bundle online security and tech support

## Project files
| File | Purpose |
|---|---|
| `Telco_Customer_Churn_Project.ipynb` | Full analysis and modelling notebook |
| `app.py` | Streamlit app |
| `predict.py` | Prediction logic used by the app |
| `churn_model.pkl` | Saved model pipeline (preprocessing + model + threshold) |
| `requirements.txt` | Python packages |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset |

## Run the app locally
Python 3.11 or newer is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model was trained with scikit-learn 1.7.2, so keep the pinned versions in `requirements.txt`.

## Limitations
- The data is a single snapshot, and the patterns are associations, not proven causes.
- Business numbers (offer cost, customer value) are assumptions and should be replaced with real values.
- Model quality plateaus near ROC-AUC 0.85. More data (usage, complaints, call-centre contacts) would help more than a more complex model.