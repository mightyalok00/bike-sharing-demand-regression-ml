# Bike Sharing Demand — Professional End-to-End Regression Project

Production-style machine-learning project for the Kaggle Bike Sharing Demand dataset.

## Project structure

```text
bike-sharing-demand-regression-ml/
├── app.py
├── train_model.py
├── generate_submission.py
├── requirements.txt
├── Dockerfile
├── railway.toml
├── .dockerignore
├── .gitignore
├── README.md
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── notebooks/
│   └── bike_sharing_demand_18_questions.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── features.py
    ├── modeling.py
    └── evaluation.py
```

## 18 questions covered

1. Dataset structure and column meaning
2. YData Profiling and data quality
3. Demand patterns by season, weather, calendar and hour
4. Correlation, nonlinear relationships and outliers
5. Casual vs registered user behavior
6. Data cleaning and leakage prevention
7. Datetime/calendar feature engineering
8. LabelEncoder, OneHotEncoder and OrdinalEncoder
9. Encoding/decoding with inverse_transform
10. StandardScaler vs MinMaxScaler
11. Linear Regression baseline
12. Polynomial Regression
13. Ridge, Lasso and Elastic Net
14. Decision Tree, Random Forest and Gradient Boosting Regression
15. Accuracy/generalization/complexity comparison
16. GridSearchCV and RandomizedSearchCV
17. Pipeline and ColumnTransformer
18. Joblib serialization and Streamlit deployment

## Local data

Expected files:

```text
D:\Bike\train.csv
D:\Bike\test.csv
D:\Bike\sampleSubmission.csv
```

The raw Kaggle CSV files are intentionally excluded from Git.

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py --train "D:\Bike\train.csv"
python generate_submission.py --test "D:\Bike\test.csv" --sample "D:\Bike\sampleSubmission.csv"
streamlit run app.py
```

## Models

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Tuned Random Forest
- Tuned Gradient Boosting

The training workflow uses leakage-safe preprocessing, 5-fold cross-validation, a holdout set, log1p/expm1 target transformation, GridSearchCV and RandomizedSearchCV.

## Deployment

The repository includes Docker and Railway configuration. Train the model first and commit `models/final_model.joblib` before deploying the Streamlit application.
