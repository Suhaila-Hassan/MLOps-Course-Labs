from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict():
    sample_input = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()

