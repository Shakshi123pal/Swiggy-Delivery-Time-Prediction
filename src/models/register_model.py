import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging

# ------------------- Logger Setup -------------------
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# ------------------- Dagshub Authentication -------------------
DAGSHUB_TOKEN = "bfc83f7bf116c4e2d4e1a5f3ebc2562940f96afd"
dagshub.auth.add_app_token(DAGSHUB_TOKEN)

# Initialize Dagshub repo tracking
dagshub.init(
    repo_owner="Shakshi123pal",
    repo_name="Swiggy-Delivery-Time-Prediction",
    mlflow=True
)

# Set Dagshub tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Shakshi123pal/Swiggy-Delivery-Time-Prediction.mlflow"
)

# ------------------- Helper Function -------------------
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

# ------------------- Main Script -------------------
if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    run_info_path = root_path / "run_information.json"
    run_info = load_model_information(run_info_path)

    run_id = run_info["run_id"]
    model_name = run_info["model_name"]

    model_registry_path = f"runs:/{run_id}/{model_name}"

    # Use MlflowClient with Dagshub directly
    registry_client = MlflowClient()

    # Check if model already exists, else create
    try:
        registry_client.get_registered_model(model_name)
        logger.info(f"Model '{model_name}' already exists on Dagshub.")
    except Exception:
        registry_client.create_registered_model(model_name)
        logger.info(f"Created new registered model '{model_name}' on Dagshub.")

    # Register a new model version
    model_version = registry_client.create_model_version(
        name=model_name,
        source=model_registry_path,
        run_id=run_id
    )
    logger.info(f"New version {model_version.version} created for model '{model_name}'")

    # Transition to "Staging"
    registry_client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )
    logger.info(f"Model '{model_name}' pushed to Staging stage on Dagshub")
