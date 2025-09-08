import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn
from pathlib import Path

# ------------------- Load Model + Preprocessor -------------------
root_path = Path(__file__).parent
model_path = root_path / "models" / "delivery_time_model.pkl"
preprocessor_path = root_path / "models" / "preprocessor.joblib"   # yeh training ke time save karna tha

preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)

# ------------------- FastAPI App -------------------
app = FastAPI(title="Swiggy Delivery Time Prediction API")

class InputData(BaseModel):
    age: float
    ratings: float
    weather: str
    traffic: str
    vehicle_condition: int
    type_of_order: str
    type_of_vehicle: str
    multiple_deliveries: float
    festival: str
    city_type: str
    is_weekend: int
    pickup_time_minutes: float
    order_time_of_day: str
    distance: float
    distance_type: str

@app.get("/")
def home():
    return {"message": "🚀 Swiggy Delivery Time Prediction API is running"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])

    # Apply preprocessing before prediction
    X_transformed = preprocessor.transform(df)
    prediction = model.predict(X_transformed)

    return {"predicted_delivery_time": float(prediction[0])}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
