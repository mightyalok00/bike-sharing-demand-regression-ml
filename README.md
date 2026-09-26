# Bike Sharing Demand — Professional End-to-End Regression Project

Production-style machine-learning project for the Kaggle Bike Sharing Demand dataset.

## 🚲 Live application

The Streamlit app provides an interactive BikePulse demand predictor with:
- 🎛️ Sidebar filters
- ⚡ Quick demand scenarios
- 📅 Date/time controls
- 🌦️ Weather and season controls
- 🌡️ Temperature, humidity and wind controls
- 🔮 Hourly rental prediction
- 📊 Demand-level indicators
- 🧠 Model information and leakage-protection notes

## Project structure

```text
bike-sharing-demand-regression-ml/
├── app.py
├── train_model.py
├── generate_submission.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── railway.toml
├── .dockerignore
├── .gitignore
├── README.md
├── data/
├── models/
├── reports/
├── notebooks/
└── src/
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

## 🧪 Local setup

For the Streamlit application:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For full notebook/EDA work:

```powershell
pip install -r requirements-dev.txt
```

Train and generate the model:

```powershell
python train_model.py --train "D:\Bike\train.csv"
python generate_submission.py --test "D:\Bike\test.csv" --sample "D:\Bike\sampleSubmission.csv"
streamlit run app.py
```

## 🤖 Models

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

## ☁️ Streamlit Community Cloud

Use the GitHub repository as the deployment source and set the main file to:

```text
app.py
```

Streamlit deployment intentionally uses the lightweight `requirements.txt`. The notebook/EDA packages are kept in `requirements-dev.txt` so they do not slow down or break app dependency installation.

Before deploying, ensure `models/final_model.joblib` is available to the application environment.

## 🐳 Docker / Railway

Build locally:

```bash
docker build -t bike-sharing-demand .
docker run -p 8501:8501 bike-sharing-demand
```

The Dockerfile automatically respects Railway's `PORT` environment variable.

## 🔐 Deployment dependency design

`requirements.txt` contains only packages required to run the deployed prediction app.

`requirements-dev.txt` adds:
- Matplotlib
- Seaborn
- YData Profiling
- Jupyter
- nbformat

This keeps Streamlit Cloud and Docker builds lighter and more reliable.
