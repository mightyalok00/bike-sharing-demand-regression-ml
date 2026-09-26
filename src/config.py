from pathlib import Path
import os

DEFAULT_WINDOWS_TRAIN = r"D:Bike	rain.csv"
DEFAULT_WINDOWS_TEST = r"D:Bike	est.csv"
DEFAULT_WINDOWS_SAMPLE = r"D:BikesampleSubmission.csv"

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "final_model.joblib"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
HOLDOUT_PATH = REPORT_DIR / "holdout_metrics.csv"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TRAIN_PATH = os.getenv("BIKE_TRAIN_PATH", DEFAULT_WINDOWS_TRAIN)
TEST_PATH = os.getenv("BIKE_TEST_PATH", DEFAULT_WINDOWS_TEST)
SAMPLE_PATH = os.getenv("BIKE_SAMPLE_PATH", DEFAULT_WINDOWS_SAMPLE)
