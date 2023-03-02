from fastapi import FastAPI
from fastapi.testclient import TestClient
import os
from main import app


tests_directory_path = os.path.dirname(__file__)
client = TestClient(app)

def test_create_predictions():
    response = client.post("/predict/furnitureimage", files = {"file":open(os.path.join(tests_directory_path, 'test_data/image_test.png'), "rb")})
    assert response.status_code == 200
    assert response.json() == {
    "model-prediction": "Sofa",
    "model-prediction-confidence-score": 64.67121243476868
}
    
