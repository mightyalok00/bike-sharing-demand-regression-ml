from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, REPORT_DIR, SAMPLE_PATH, TEST_PATH
from src.data import basic_clean, load_csv, validate_test
from src.features import prepare_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default=TEST_PATH)
    p.add_argument("--sample", default=SAMPLE_PATH)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--output", default=str(REPORT_DIR / "submission.csv"))
    args = p.parse_args()

    test = basic_clean(load_csv(args.test))
    validate_test(test)

    bundle = joblib.load(args.model)
    model = bundle["model"]

    X_test = prepare_features(test)
    prediction = np.maximum(0, np.expm1(model.predict(X_test)))
    prediction = np.rint(prediction).astype(int)

    sample = load_csv(args.sample)
    if "datetime" not in sample.columns:
        raise ValueError("sampleSubmission.csv must contain a datetime column.")

    submission = pd.DataFrame(
        {
            "datetime": test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "count": prediction,
        }
    )

    if list(sample.columns) == ["datetime", "count"]:
        submission = submission[["datetime", "count"]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved submission: {args.output}")


if __name__ == "__main__":
    main()
