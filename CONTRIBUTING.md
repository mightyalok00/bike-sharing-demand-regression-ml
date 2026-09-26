# Contributing to BikePulse

Thanks for contributing to **BikePulse**.

## Development setup

Use Python 3.12+ and create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Before opening a pull request

Run:

```powershell
ruff check src bootstrap_model.py train_model.py tests
pytest -q --cov=src --cov-report=term-missing
python -m compileall -q src bootstrap_model.py train_model.py
```

## ML contribution guidelines

- Keep target leakage out of prediction features.
- Reuse the existing feature-engineering pipeline where possible.
- Keep preprocessing inside Scikit-learn pipelines.
- Set deterministic random states when supported.
- Add or update tests for behavior changes.
- Do not commit Kaggle dataset files or generated secrets.
- Keep Streamlit deployment compatibility in mind.

## Pull requests

Please include:

- What changed
- Why it changed
- Tests run locally
- Any model or metric impact
- Screenshots for meaningful Streamlit UI changes

Small, focused pull requests are preferred.
