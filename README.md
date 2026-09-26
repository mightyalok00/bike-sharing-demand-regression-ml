# 🚲 BikePulse — Bike Sharing Demand Prediction

<p align="center">
  <strong>End-to-end regression machine learning project for hourly bike rental demand forecasting.</strong><br/>
  Built with Python, Scikit-learn and Streamlit.
</p>

<p align="center">
  <a href="https://bike-sharing-demand-regression-ml.streamlit.app/"><strong>🚀 Live Demo</strong></a> ·
  <a href="https://github.com/mightyalok00/bike-sharing-demand-regression-ml"><strong>📦 GitHub</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/github/actions/workflow/status/mightyalok00/bike-sharing-demand-regression-ml/ci.yml?branch=main&style=for-the-badge&label=CI" alt="CI"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

---

## 📌 Overview

**BikePulse** is a production-oriented machine learning project based on the **Kaggle Bike Sharing Demand** dataset.

The project demonstrates a complete regression workflow:

**data validation → feature engineering → leakage prevention → preprocessing → model comparison → cross-validation → hyperparameter tuning → model serialization → interactive prediction**

The deployed application lets users configure calendar, weather and environmental conditions and generate an hourly bike-demand prediction.

## 🚀 Live Application

### BikePulse — Interactive Demand Predictor

**[Launch BikePulse](https://bike-sharing-demand-regression-ml.streamlit.app/)**

The Streamlit application provides:

- 🎛️ Interactive prediction controls
- ⚡ Quick scenario presets
- 🧠 Regression model selection
- 📅 Date and time features
- 🌦️ Weather and season scenarios
- 🌡️ Temperature, humidity and wind inputs
- 🔮 Hourly demand prediction
- 📊 Demand-level interpretation
- 🛡️ Target-leakage protection

> **Deployment:** Streamlit Community Cloud  
> **Main file:** `app.py`

---

## 🎯 Project Objective

Predict the number of bike rentals for a given hour using information available before the rental outcome occurs.

The project specifically addresses **target leakage** by excluding post-outcome variables such as:

- `count`
- `casual`
- `registered`

This makes the prediction workflow more representative of a real forecasting scenario.

---

## 🧠 Regression Models

The project compares **8 regression approaches**:

| Model | Purpose |
|---|---|
| Linear Regression | Baseline linear relationship |
| Polynomial Regression | Captures nonlinear relationships |
| Ridge Regression | L2-regularized linear regression |
| Lasso Regression | L1-regularized linear regression |
| Elastic Net | L1 + L2 regularization |
| Decision Tree Regression | Nonlinear rule-based modeling |
| Random Forest Regression | Bagged tree ensemble |
| Gradient Boosting Regression | Sequential boosted-tree ensemble |

The training workflow additionally performs hyperparameter search for:

- **Random Forest Regression**
- **Gradient Boosting Regression**

---

## 🔬 Machine Learning Pipeline

```text
                 Kaggle / Local Data
                         │
                         ▼
                 Data Validation
                         │
                         ▼
                  Data Cleaning
                         │
                         ▼
              Datetime Feature Engineering
                         │
                         ▼
                Leakage Prevention
                         │
                         ▼
             ColumnTransformer Pipeline
                         │
                         ▼
               Regression Models
                         │
                         ▼
                5-Fold Cross-Validation
                         │
                         ▼
              Hyperparameter Tuning
                         │
                         ▼
                 Holdout Evaluation
                         │
                         ▼
              Joblib Model Serialization
                         │
                         ▼
                 Streamlit Prediction
```

---

## 🏗️ Architecture

BikePulse separates the application layer, machine-learning pipeline and reusable data/model components.

```text
                    ┌─────────────────────────┐
                    │   Kaggle Bike Dataset   │
                    │ train.csv / test.csv    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   src/data.py            │
                    │ Load • Clean • Validate  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   src/features.py        │
                    │ Datetime • Cyclical      │
                    │ Leakage-safe features   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   src/modeling.py        │
                    │ Preprocessing pipelines │
                    │ Regression estimators   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ train_model.py   │      │ tests/           │
          │ CV • tuning •    │      │ Automated checks │
          │ evaluation       │      └──────────────────┘
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ models/*.joblib  │
          │ reports/*.csv    │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │     app.py       │
          │ Streamlit UI     │
          │ Prediction +     │
          │ Analytics        │
          └──────────────────┘
```

### Architecture principles

- **Separation of concerns** — data, features, modeling and evaluation live in reusable modules.
- **Leakage-safe training** — post-outcome target fields are removed before modeling.
- **Pipeline-based preprocessing** — transformations are kept with the estimator workflow.
- **Reproducible training** — fixed random state and documented commands.
- **Artifact isolation** — generated models and reports are kept outside the source modules.
- **Testable components** — core data, feature, evaluation and modeling utilities have automated tests.

---

## ☁️ Deployment

### Streamlit Community Cloud

The production-facing demo is deployed as a **Streamlit Community Cloud** application.

```text
GitHub main branch
       │
       ▼
Streamlit Community Cloud
       │
       ▼
      app.py
       │
       ├── src/features.py
       ├── src/modeling.py
       └── saved model artifact
       │
       ▼
Interactive BikePulse web app
```

**Entry point:** `app.py`

**Live application:** [BikePulse](https://bike-sharing-demand-regression-ml.streamlit.app/)

### Continuous Integration

GitHub Actions validates the repository independently of the Streamlit deployment:

```text
Push / Pull Request / Manual Run
                │
                ▼
        Install dependencies
                │
                ▼
             Ruff
                │
                ▼
       Pytest + coverage
                │
                ▼
       Python compilation
```

This keeps deployment and code-quality validation separate: Streamlit serves the application, while GitHub Actions checks the codebase.

---

## 🧩 Feature Engineering

The project transforms the original datetime information into predictive calendar and cyclical features.

### Calendar features

- Year
- Month
- Day
- Hour
- Weekday
- Week of year
- Weekend indicator
- Rush-hour indicator
- Peak-hour indicator

### Cyclical features

- Hour sine/cosine encoding
- Month sine/cosine encoding
- Weekday sine/cosine encoding

These features help regression models represent recurring temporal patterns.

---

## 📊 Model Evaluation

Models are evaluated using:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of determination
- **Cross-validation RMSE**
- **Holdout validation**

The training workflow also uses a `log1p` target transformation and `expm1` inverse transformation to model the skewed demand target.

### Reproducible benchmark

The repository intentionally does **not** invent benchmark numbers. Run the training pipeline on the Kaggle dataset to generate:

- `reports/model_comparison.csv`
- `reports/holdout_metrics.csv`
- `reports/feature_importance.csv`

Once a benchmark is generated, it can be added to this section with the exact training configuration used.

---

## 📊 Interactive Analytics

The Streamlit application includes an **Analytics** view for inspecting the active regression model.

It can show:

- Top model features for estimators exposing coefficients or tree feature importance
- A feature-importance chart
- Training benchmark tables when generated reports are available
- Active model and target-transformation metadata

Generate reproducible benchmark reports locally with:

```powershell
python train_model.py --train "D:\Bike\train.csv"
```

The generated `reports/model_comparison.csv` is intentionally not committed automatically because it depends on the local training dataset and configuration.

---
## 🧪 Data Science Topics

The accompanying notebook covers **18 practical topics**:

1. Dataset structure and column interpretation
2. YData Profiling and data-quality analysis
3. Demand patterns by season, weather, calendar and hour
4. Correlation, nonlinear relationships and outliers
5. Casual vs. registered user behavior
6. Data cleaning and leakage prevention
7. Datetime and calendar feature engineering
8. LabelEncoder, OneHotEncoder and OrdinalEncoder
9. Encoding and `inverse_transform`
10. StandardScaler vs. MinMaxScaler
11. Linear Regression
12. Polynomial Regression
13. Ridge, Lasso and Elastic Net
14. Decision Tree, Random Forest and Gradient Boosting
15. Model accuracy, generalization and complexity
16. GridSearchCV and RandomizedSearchCV
17. Pipeline and ColumnTransformer
18. Joblib serialization and deployment

---

## 📁 Project Structure

```text
bike-sharing-demand-regression-ml/
│
├── app.py                    # Streamlit application
├── train_model.py            # Model training pipeline
├── generate_submission.py    # Kaggle submission generator
├── bootstrap_model.py        # Model/data bootstrap utilities
│
├── src/
│   ├── config.py             # Project configuration
│   ├── data.py               # Loading, cleaning and validation
│   ├── features.py           # Feature engineering
│   ├── modeling.py           # Model and pipeline definitions
│   ├── evaluation.py         # Metrics and feature importance
│   └── __init__.py
│
├── tests/                    # Automated test suite
├── notebooks/                # Exploratory analysis notebook
├── data/                     # Dataset instructions
├── models/                   # Generated model artifacts
├── reports/                  # Generated evaluation/submission files
│
├── .github/workflows/       # GitHub Actions CI
├── .streamlit/              # Streamlit configuration
│
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Development/EDA/test dependencies
├── pyproject.toml            # Ruff/project configuration
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Local Installation

### 1. Clone the repository

```powershell
git clone https://github.com/mightyalok00/bike-sharing-demand-regression-ml.git
cd bike-sharing-demand-regression-ml
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

For the Streamlit application:

```powershell
pip install -r requirements.txt
```

For development, notebooks and testing:

```powershell
pip install -r requirements-dev.txt
```

---

## ▶️ Run the Streamlit App

Start the application locally:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 🏋️ Train the Models

The training pipeline expects the Kaggle competition files to be available locally.

Example:

```powershell
python train_model.py --train "D:\Bike\train.csv"
```

The training process includes:

- Data validation
- Data cleaning
- Leakage-safe feature engineering
- Train/holdout split
- 5-fold cross-validation
- Multiple regression models
- `log1p` target transformation
- RandomizedSearchCV
- GridSearchCV
- Holdout evaluation
- Feature-importance analysis
- Joblib serialization

Generated artifacts are written to:

```text
models/
reports/
```

---

## 📦 Generate a Kaggle Submission

After training a model:

```powershell
python generate_submission.py --test "D:\Bike\test.csv" --sample "D:\Bike\sampleSubmission.csv"
```

The generated submission is written to:

```text
reports/submission.csv
```

Raw Kaggle CSV files are intentionally **not committed to this repository**.

---

## 🗂️ Dataset

The project uses the **Kaggle Bike Sharing Demand** competition dataset.

Expected local files:

```text
D:\Bike\train.csv
D:\Bike\test.csv
D:\Bike\sampleSubmission.csv
```

Dataset source:

**[Kaggle — Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand/data)**

The competition dataset is subject to Kaggle's applicable competition rules. The MIT License applies only to the original code in this repository.

---

## 🧪 Testing & Code Quality

The repository includes automated tests covering:

- Datetime and cyclical feature engineering
- Target-leakage removal
- Input schema validation
- Data cleaning
- Regression metrics
- Model fitting and prediction
- Polynomial regression

Run the test suite:

```powershell
pytest -q --cov=src --cov-report=term-missing
```

Run Ruff:

```powershell
ruff check src bootstrap_model.py train_model.py tests
```

---

## 🔄 Continuous Integration

GitHub Actions runs the same lint, test and compilation checks on pushes and pull requests. The workflow also includes `workflow_dispatch`, so you can manually run it from **GitHub → Actions → CI → Run workflow**.

---

## 🧪 Local Quality Checks

The repository keeps quality checks lightweight and reproducible locally. Run them before committing:

```powershell
ruff check src bootstrap_model.py train_model.py generate_submission.py tests
pytest -q --cov=src --cov-report=term-missing
python -m compileall -q src bootstrap_model.py train_model.py generate_submission.py
```

---

## 🔐 Engineering Practices

This project follows several production-oriented practices:

- **Modular architecture** — reusable functionality is separated into `src/`
- **Pipeline-based preprocessing** — preprocessing and models are composed with Scikit-learn pipelines
- **Leakage prevention** — outcome-derived features are excluded from prediction inputs
- **Reproducibility** — `RANDOM_STATE = 42` is used where supported
- **Dependency separation** — runtime and development dependencies are maintained separately
- **Automated testing** — core ML utilities are covered by tests
- **Local quality checks** — linting, testing and compilation commands are documented
- **Artifact separation** — generated models and reports are separated from source code

---

## 🧭 Roadmap

- [x] Exploratory data analysis
- [x] Feature engineering
- [x] Multiple regression algorithms
- [x] Cross-validation
- [x] Hyperparameter tuning
- [x] Joblib serialization
- [x] Streamlit application
- [x] Automated test suite
- [x] MIT license
- [ ] Add verified benchmark table from a reproducible training run
- [x] Add prediction analytics / model explainability
- [ ] Add production monitoring

---

## 🤝 Contributing

Contributions are welcome. Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for the development workflow, testing expectations and pull-request checklist.

---

## 👤 Author

### Alok Agarwal

**Data Science · Machine Learning · Python · SEO & Digital Marketing**

- **GitHub:** [@mightyalok00](https://github.com/mightyalok00)
- **LinkedIn:** [Alok Agarwal](https://www.linkedin.com/in/alok-agarwal-seo-digital-marketing)

---

## 📄 License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with Python · Scikit-learn · Streamlit</strong><br/>
  <sub>BikePulse — Predicting hourly bike-sharing demand with machine learning.</sub>
</p>
