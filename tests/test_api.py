from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_normal():
    response = client.post(
        "/predict",
        json={
            "type": "M",
            "air_temp_k": 298.1,
            "process_temp_k": 308.6,
            "rotational_speed_rpm": 1551,
            "torque_nm": 42.8,
            "tool_wear_min": 0,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["failure_prediction"] == 0
    assert body["status"] == "OK"


def test_predict_critical():
    response = client.post(
        "/predict",
        json={
            "type": "L",
            "air_temp_k": 300.5,
            "process_temp_k": 309.0,
            "rotational_speed_rpm": 1350,
            "torque_nm": 68.5,
            "tool_wear_min": 215,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["failure_prediction"] == 1
    assert body["status"] == "CRITICAL"


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={
            "type": "X",
            "air_temp_k": -50.0,
            "process_temp_k": 308.6,
            "rotational_speed_rpm": 1551,
            "torque_nm": 42.8,
            "tool_wear_min": 0,
        },
    )
    assert response.status_code == 422