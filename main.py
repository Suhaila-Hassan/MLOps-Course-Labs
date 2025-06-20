from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
import numpy as np
import pickle
import logging
import uvicorn
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
MODEL_PATH = "models/RandomForest.pkl"
TRANSFORMER_PATH = "models/column_transformer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load transformer
with open(TRANSFORMER_PATH, "rb") as f:
    column_transformer = pickle.load(f)

cat_cols = ["Geography", "Gender"]
num_cols = ["CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", 
            "EstimatedSalary"]

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Defines a location in dictionary for user input
class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Welcome"}

@app.get("/health")
def health():
    logger.info("System Healthy")
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CustomerData):
    logger.info(f"Received request: {data}")
    try:
        input_df = pd.DataFrame([data.dict()])
        logger.info(f"Input DataFrame:\n{input_df}")

        transformed_input = column_transformer.transform(input_df)
        prediction = model.predict(transformed_input)[0]

        logger.info(f"Prediction result: {prediction}")
        return {"churn_prediction": int(prediction)}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

