from fastapi.testclient import TestClient

from src.serving.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "transformer_dot_L1_seed42"


def test_prediction():
    
    payload = {
        "adgroup_id": 775579,
        "cate_id": 6261,
        "brand": 270915,
        "price": 0.14666666090488434,
        "cate_history": [6261] * 50,
        "brand_history": [270915] * 50,
        "btag_history": [1] * 50,
    }

    assert len(payload["cate_history"]) == 50
    assert len(payload["brand_history"]) == 50
    assert len(payload["btag_history"]) == 50

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "click_probability" in data
    assert "prediction" in data
    assert "model" in data

    assert 0.0 <= data["click_probability"] <= 1.0
    assert data["prediction"] in [0, 1]
    assert data["model"] == "transformer_dot_L1_seed42"


def test_invalid_price():
    payload = {
        "adgroup_id": 775579,
        "cate_id": 6261,
        "brand": 270915,
        "price": -10,
        "cate_history": [],
        "brand_history": [],
        "btag_history": [],
    }
    print("cate:", len(payload["cate_history"]))
    print("brand:", len(payload["brand_history"]))
    print("btag:", len(payload["btag_history"]))

    response = client.post(
        "/predict",
        json=payload,
    )

    print(response.json())

    assert response.status_code == 200