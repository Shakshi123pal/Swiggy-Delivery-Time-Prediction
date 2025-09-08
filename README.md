🚀 Swiggy Delivery Time Prediction

This project predicts food delivery time (in minutes) for Swiggy orders using machine learning and MLOps best practices.

📂 Project Structure

data/ → Raw and processed datasets

src/ → Scripts for data cleaning, preprocessing, training, evaluation, and model registration

models/ → Saved preprocessor + ML model

app.py → FastAPI application for real-time predictions

dvc.yaml → DVC pipeline for reproducibility

register_info.json → Model registry metadata

⚙️ Tech Stack

Python, Pandas, Scikit-learn

DVC (Data Version Control)

MLflow (experiment tracking & model registry)

FastAPI (REST API deployment)

Joblib (model persistence)

🛠️ Pipeline (DVC Stages)

Data Cleaning → remove invalid values (age < 18, ratings > 5, etc.)

Preprocessing → encoding categorical + scaling numerical features

Model Training → train ML model (Random Forest / XGBoost)

Evaluation → log metrics with MLflow

Register Model → store preprocessor + model artifacts

API Service → FastAPI endpoint for predictions

🚀 How to Run
# Clone repo
git clone <https://github.com/Shakshi123pal/Swiggy-Delivery-Time-Prediction>
cd Swiggy-Delivery-Time-Prediction

# Reproduce pipeline
dvc repro

# Start API
uvicorn app:app --reload

🔮 Future Work

Deploying the API on cloud platforms (AWS / Render / Streamlit Cloud)

Building a real-time monitoring system for model performance

Experimenting with deep learning models for better accuracy

Containerizing with Docker + CI/CD pipelines for production-level deployment
