# Contributing to BikePulse

Thanks for contributing to **BikePulse**.

## Development setup

Use Python 3.12+:

~~~powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
~~~

## Before opening a pull request

~~~powershell
ruff check src bootstrap_model.py train_model.py generate_submission.py tests
pytest -q --cov=src --cov-report=term-missing
python -m compileall -q src bootstrap_model.py train_model.py generate_submission.py
~~~

## ML contribution guidelines

- Keep target leakage out of prediction features.
- Reuse the existing feature-engineering pipeline.
- Keep preprocessing inside Scikit-learn pipelines.
- Use deterministic random states where supported.
- Add or update tests when behavior changes.
- Do not commit Kaggle CSV files, model binaries or secrets.
- Keep Streamlit deployment compatibility in mind.
- If model metrics change, explain why and regenerate the committed evaluation report.

## Pull requests

Include:

- What changed
- Why it changed
- Tests run
- Model/metric impact, if applicable
- Screenshots for meaningful Streamlit UI changes

Keep changes focused and consistent with the repository architecture.
