# test_api.py
from fastapi.testclient import TestClient
from src.deployment.api import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"sample":[0.32,0.45,0.67]})
    assert response.status_code == 200
    assert "DON Concentration" in response.json()
