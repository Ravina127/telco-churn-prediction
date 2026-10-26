# Telco Customer Churn Prediction

Predicts which telecom customers are likely to leave (churn), explains the main drivers of churn, and estimates the business value of targeting high-risk customers.

**Live app:** _add your Streamlit link here after deploying_

## Problem
Winning a new customer costs far more than keeping an existing one. The goal is to find at-risk customers early, so the company can offer them the right help before they leave.

## Data
Telco Customer Churn dataset: 7,043 customers, 21 columns (demographics, account details, services, and the target `Churn`). About 26.5% of customers churned.

## Approach
1. Data cleaning (fixed `TotalCharges`, encoded the target)
2. Exploratory data analysis
3. Feature engineering (`NumAddOns`, `AvgMonthlySpend`)
4. Model comparison with 5-fold cross-validation (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
5. Hyperparameter tuning, evaluation on an untouched test set
6. Threshold selection, permutation importance, business impact

## Key results
| Item | Result |
|---|---|
| Overall churn rate | 26.5% |
| Best model | Gradient Boosting (tuned) |
| Test ROC-AUC | 0.845 |
| Threshold 0.40 | Catches 64% of churners with 60% precision |
| Top 20% riskiest customers | Contain about half of all churners (2.5x better than random) |

**Top churn drivers:** contract type, tenure, internet service, tech support, online security.

**Highest-risk group:** month-to-month contract + fibre optic (55% churn).

## Recommendations
- Move month-to-month customers to longer contracts
- Add onboarding support in the first 12 months
- Review fibre-optic quality and pricing
- Encourage automatic payments
- Bundle online security and tech support

## Project files
| File | Purpose |
|---|---|
| `Telco_Customer_Churn_Project.ipynb` | Full analysis and modelling notebook |
| `app.py` | Streamlit app |
| `predict.py` | Prediction logic used by the app |
| `churn_model.pkl` | Saved model pipeline (preprocessing + model + threshold) |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset |

## Run the app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
- The data is a single snapshot, and the patterns are associations, not proven causes.
- Business numbers (offer cost, customer value) are assumptions.
- Model quality plateaus near ROC-AUC 0.85. More data (usage, complaints) would help more than a more complex model.

## Author
Your Name
