import joblib
import logging
from pathlib import Path
import json

# ------------------- Logger -------------------
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_model(model_path: Path):
    return joblib.load(model_path)


def save_final_model(model, final_model_path: Path):
    joblib.dump(model, final_model_path)
    logger.info(f"✅ Model registered at {final_model_path}")


def save_register_info(info_path: Path, model_name: str, final_model_path: Path, metrics_path: Path):
    # evaluation ke results bhi link karenge
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    register_info = {
        "model_name": model_name,
        "model_path": str(final_model_path),
        "metrics": metrics
    }

    with open(info_path, "w") as f:
        json.dump(register_info, f, indent=4)

    logger.info(f"📄 Register info saved at {info_path}")


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    trained_model_path = root_path / "models" / "model.joblib"   # from train.py
    final_model_path = root_path / "models" / "delivery_time_model.pkl"  # final serving model
    metrics_path = root_path / "run_information.json"             # from evaluation.py
    register_info_path = root_path / "register_info.json"

    # Load trained model
    model = load_model(trained_model_path)

    # Save as final serving model
    save_final_model(model, final_model_path)

    # Save registration info
    save_register_info(register_info_path, "delivery_time_prediction_model", final_model_path, metrics_path)
