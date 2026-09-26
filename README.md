# 🚲 Bike Sharing Demand — End-to-End Regression & Deployment

<p align="center">
  <strong>Production-style machine learning project for hourly bike rental demand forecasting</strong><br/>
  Kaggle Bike Sharing Demand · Scikit-learn · Streamlit · FastAPI · Docker
</p>

<p align="center">
  <a href="https://bike-sharing-demand-regression-ml.streamlit.app/">Live Streamlit App</a> ·
  <a href="https://github.com/mightyalok00/bike-sharing-demand-regression-ml">GitHub Repository</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-Container-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License"/>
</p>

---

## 📌 Project Overview

This repository is an end-to-end regression machine learning project built around the **Kaggle Bike Sharing Demand** competition.

The project covers data exploration, feature engineering, model comparison, hyperparameter tuning, model serialization, interactive prediction, REST API development, and containerized deployment.

### 🎯 Objective

Predict hourly bike rental demand using calendar, weather, and environmental features while preventing target leakage.

## 🌐 Live Demo

### BikePulse — Interactive Bike Demand Predictor

**[Launch the Streamlit application](https://bike-sharing-demand-regression-ml.streamlit.app/)**

The dashboard includes:

- 🎛️ Interactive prediction filters
- ⚡ Quick-demand scenarios
- 📅 Date/time controls
- 🌦️ Season and weather controls
- 🌡️ Temperature, humidity and wind controls
- 🧠 Regression-model selection
- 🔮 Hourly demand prediction
- 📊 Demand-level interpretation
- 🌑 Dark-mode interface
- 🛡️ Leakage-protection notes

## 📁 Project Structure

```text
bike-sharing-demand-regression-ml/
├── app.py
├── train_model.py
├── generate_submission.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── railway.toml
├── api.py
├── bootstrap_model.py
├── .streamlit/
├── .dockerignore
├── .gitignore
├── README.md
├── LICENSE
├── data/
├── models/
├── reports/
├── notebooks/
└── src/
```

## 🧠 Machine Learning Models

The project implements **8 regression model types**:

| Model | Role |
|---|---|
| Linear Regression | Baseline linear model |
| Polynomial Regression | Nonlinear relationships |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization |
| Elastic Net | Combined L1 + L2 regularization |
| Decision Tree Regression | Nonlinear rule-based model |
| Random Forest Regression | Bagged tree ensemble |
| Gradient Boosting Regression | Boosted tree ensemble |

The training workflow also includes tuned Random Forest and Gradient Boosting configurations.

### Modeling workflow

```text
Raw Data
   ↓
EDA & Data Quality
   ↓
Datetime / Calendar Features
   ↓
Leakage Prevention
   ↓
Preprocessing
   ↓
Cross-Validation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Model Serialization
   ↓
Streamlit + FastAPI Deployment
```

## 🔬 Data Science Topics Covered

The accompanying notebook covers **18 practical topics**:

1. Dataset structure and column meaning
2. YData Profiling and data quality
3. Demand patterns by season, weather, calendar and hour
4. Correlation, nonlinear relationships and outliers
5. Casual vs. registered user behavior
6. Data cleaning and leakage prevention
7. Datetime/calendar feature engineering
8. LabelEncoder, OneHotEncoder and OrdinalEncoder
9. Encoding/decoding with `inverse_transform`
10. StandardScaler vs MinMaxScaler
11. Linear Regression
12. Polynomial Regression
13. Ridge, Lasso and Elastic Net
14. Decision Tree, Random Forest and Gradient Boosting
15. Model accuracy, generalization and complexity
16. GridSearchCV and RandomizedSearchCV
17. Pipeline and ColumnTransformer
18. Joblib serialization and deployment

## 🧩 Feature Engineering

The project derives calendar and cyclical features such as:

- Year, month, day and hour
- Weekday and week-of-year
- Weekend indicator
- Rush-hour and peak-hour indicators
- Cyclical hour features
- Cyclical month features
- Cyclical weekday features

### Leakage protection

Post-outcome fields such as `count`, `casual`, and `registered` are excluded from prediction inputs.

## 💾 Dataset

Expected local files:

```text
D:\Bike\train.csv
D:\Bike\test.csv
D:\Bike\sampleSubmission.csv
```

The raw Kaggle CSV files are intentionally **not committed to GitHub**.

## 🧪 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/mightyalok00/bike-sharing-demand-regression-ml.git
cd bike-sharing-demand-regression-ml
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install runtime dependencies

```powershell
pip install -r requirements.txt
```

For notebook and EDA work:

```powershell
pip install -r requirements-dev.txt
```

## 🏋️ Training & Submission

Train the models:

```powershell
python train_model.py --train "D:\Bike\train.csv"
```

Generate a Kaggle-style submission:

```powershell
python generate_submission.py --test "D:\Bike\test.csv" --sample "D:\Bike\sampleSubmission.csv"
```

The training workflow includes:

- Leakage-safe preprocessing
- Train/validation split
- 5-fold cross-validation
- `log1p` target transformation
- `expm1` inverse transformation
- Model comparison
- GridSearchCV
- RandomizedSearchCV
- Feature importance analysis
- Joblib serialization

## 🎛️ Streamlit Application

Run locally:

```powershell
streamlit run app.py
```

The application supports interactive model selection and prediction controls matching the FastAPI interface.

For Streamlit Community Cloud, use:

```text
Main file: app.py
```

Runtime dependencies are intentionally kept in the lightweight `requirements.txt`.

## 🚀 FastAPI REST API

The repository contains a production-style FastAPI service.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status and service information |
| GET | `/health` | Health check |
| GET | `/models` | Available models and prediction controls |
| POST | `/predict` | Generate a bike-demand prediction |
| GET | `/docs` | Swagger / OpenAPI interface |

### API flow

```text
Client
  ↓
POST /predict
  ↓
FastAPI validation
  ↓
Feature engineering
  ↓
Selected regression model
  ↓
Prediction
  ↓
JSON response
```

The API accepts controls for:

- Date/time
- Scenario
- Regression model
- Season
- Weather
- Holiday
- Working day
- Temperature
- Feels-like temperature
- Humidity
- Wind speed

## 🐳 Docker

Build the API image:

```bash
docker build -t bike-sharing-demand .
```

Run locally:

```bash
docker run --rm -p 8000:8000 bike-sharing-demand
```

Health check:

```text
http://localhost:8000/health
```

Swagger:

```text
http://localhost:8000/docs
```

The Dockerfile is configured to use the deployment platform's `PORT` environment variable.

## ☁️ Deployment

### Streamlit

The interactive frontend is deployed separately:

**[Open BikePulse](https://bike-sharing-demand-regression-ml.streamlit.app/)**

### FastAPI / Docker

The API is Docker-ready for services such as:

- Railway
- Render Web Service
- Google Cloud Run
- Koyeb

Use a **Web Service / container service** for the FastAPI backend rather than a Static Site.

## 📊 Model Evaluation

The project evaluates models using:

- Cross-validation performance
- Holdout validation
- RMSE
- MAE
- R²
- Log-target validation where appropriate

Performance figures are intentionally not hard-coded here; they should come from a reproducible training run using the local competition dataset.

## 📦 Kaggle Submission

The repository includes `generate_submission.py`, which generates a prediction file using the Kaggle sample-submission structure.

Output:

```text
reports/submission.csv
```

when the required local inputs are available.

## 📜 Dataset & Licensing

### Original project code

The original source code and project files authored for this repository are licensed under the **MIT License**.

See [LICENSE](LICENSE).

### Kaggle dataset

This project uses the **Kaggle Bike Sharing Demand** competition dataset.

Kaggle lists the dataset as **"Subject to Competition Rules"**. The raw competition CSV files are not redistributed in this repository.

**Dataset source:** [Kaggle — Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand/data)

The MIT License applies to the project's original code and does **not** grant rights to the Kaggle dataset or other third-party materials.

## 🔐 Reproducibility & Engineering Notes

This repository intentionally separates:

- Runtime application dependencies → `requirements.txt`
- Development / notebook dependencies → `requirements-dev.txt`
- Local/raw data → excluded from Git
- Model artifacts → stored separately from raw data
- Frontend → Streamlit
- API → FastAPI
- Containerization → Docker

## 🧭 Roadmap

- [x] Exploratory data analysis
- [x] Feature engineering
- [x] Multiple regression algorithms
- [x] Cross-validation
- [x] Hyperparameter tuning
- [x] Joblib serialization
- [x] Streamlit dashboard
- [x] FastAPI REST API
- [x] Docker containerization
- [x] MIT license for original code
- [ ] Add verified benchmark table from a reproducible training run
- [ ] Add automated CI tests
- [ ] Add API integration tests
- [ ] Add production monitoring

## 👤 Author

**Alok Agarwal**

Data Science · Machine Learning · Python · SEO & Digital Marketing

- GitHub: [@mightyalok00](https://github.com/mightyalok00)
- LinkedIn: [Alok Agarwal](https://www.linkedin.com/in/alok-agarwal-seo-digital-marketing)

---

<p align="center">
  Built with Python, Scikit-learn, Streamlit, FastAPI and Docker.
</p>
