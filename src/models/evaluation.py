import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json
import os
import mlflow.pyfunc

# 🔑 Token directly daal do (abhi env ki zarurat nahi)
DAGSHUB_TOKEN = "bfc83f7bf116c4e2d4e1a5f3ebc2562940f96afd"

# ✅ Authenticate with Dagshub
dagshub.auth.add_app_token(DAGSHUB_TOKEN)

# ✅ Initialize Dagshub repo tracking
dagshub.init(
    repo_owner="Shakshi123pal",
    repo_name="Swiggy-Delivery-Time-Prediction",
    mlflow=True
)

# ✅ Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Shakshi123pal/Swiggy-Delivery-Time-Prediction.mlflow")
mlflow.set_experiment("DVC Pipeline")


TARGET = "time_taken"

# create logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(handler)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    
    return df


def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def save_model_info(save_json_path,run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path,"w") as f:
        json.dump(info_dict,f,indent=4)



if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    model_path = root_path / "models" / "model.joblib"

    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)

    X_train, y_train = make_X_and_y(train_data, TARGET)
    X_test, y_test = make_X_and_y(test_data, TARGET)

    # load trained model
    model = joblib.load(model_path)

    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring="neg_mean_absolute_error", n_jobs=-1)

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

        # log cv scores
        mlflow.log_metrics({f"CV_{i}": -score for i, score in enumerate(cv_scores)})

        #  Save model locally + log artifact to MLflow (no registry)
        model_file = "delivery_time_pred_model.pkl"
        joblib.dump(model, model_file)
        mlflow.log_artifact(model_file, artifact_path="model")

        artifact_uri = mlflow.get_artifact_uri()
    run_id = run.info.run_id
    model_name = "delivery_time_pred_model"

    save_json_path = root_path / "run_information.json"
    save_model_info(save_json_path, run_id, artifact_uri, model_name)
    logger.info("Model logged & registry info saved ")
