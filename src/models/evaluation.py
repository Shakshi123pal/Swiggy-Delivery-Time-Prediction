import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json

# ------------------- Config -------------------
DAGSHUB_TOKEN = "bfc83f7bf116c4e2d4e1a5f3ebc2562940f96afd"
TARGET = "time_taken"

# ------------------- Logger -------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# ------------------- Dagshub Auth -------------------
dagshub.auth.add_app_token(DAGSHUB_TOKEN)

dagshub.init(
    repo_owner="Shakshi123pal",
    repo_name="Swiggy-Delivery-Time-Prediction",
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/Shakshi123pal/Swiggy-Delivery-Time-Prediction.mlflow")
mlflow.set_experiment("DVC Pipeline")

# ------------------- Helpers -------------------
def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)

def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path, "w") as f:
        json.dump(info_dict, f, indent=4)

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

    # Load trained model
    model = joblib.load(model_path)

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

    # ------------------- MLflow Logging -------------------
    with mlflow.start_run() as run:
        mlflow.set_tag("model", "Food Delivery Time Regressor")
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "mean_cv_score": -(cv_scores.mean())
        })
        mlflow.log_metrics({f"CV_{i}": -score for i, score in enumerate(cv_scores)})

        # ✅ Save model in artifacts with fixed path
        artifact_subdir = "model"
        model_filename = "delivery_time_pred_model.pkl"
        joblib.dump(model, model_filename)
        mlflow.log_artifact(model_filename, artifact_path=artifact_subdir)

        artifact_uri = mlflow.get_artifact_uri()

    # ------------------- Save Run Info for Registry -------------------
    run_id = run.info.run_id
    model_name = "delivery_time_pred_model"

    save_json_path = root_path / "run_information.json"
    save_model_info(save_json_path, run_id, f"{artifact_subdir}/{model_filename}", model_name)
    logger.info("✅ Model evaluated, logged to MLflow & info saved for registry")
