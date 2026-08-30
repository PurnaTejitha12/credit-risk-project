from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "text": "This is a test credit application"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
