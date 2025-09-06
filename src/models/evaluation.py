import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json

TARGET = "time_taken"

# ------------------- Logger -------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ------------------- Helper Functions -------------------
def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)

def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def load_model(model_path: Path):
    return joblib.load(model_path)

def save_model_info(save_json_path, metrics: dict):
    with open(save_json_path, "w") as f:
        json.dump(metrics, f, indent=4)

# ------------------- Main -------------------
if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    model_path = root_path / "models" / "model.joblib"

    # Load data
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)

    X_train, y_train = make_X_and_y(train_data, TARGET)
    X_test, y_test = make_X_and_y(test_data, TARGET)

    # Load model
    model = load_model(model_path)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    mean_cv_score = -(cv_scores.mean())

    # Collect results
    results = {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "mean_cv_score": mean_cv_score,
        "cv_scores": list(-cv_scores)
    }

    logger.info(f"Evaluation results: {results}")

    # Save results locally
    save_json_path = root_path / "run_information.json"
    save_model_info(save_json_path, results)
    logger.info("Model evaluation results saved successfully")
