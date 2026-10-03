from fastapi.testclient import TestClient

from src.serving.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
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

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data
    assert data["detail"][0]["loc"] == ["body", "price"]
    assert (
        data["detail"][0]["msg"]
        == "Input should be greater than or equal to 0"
    )

    # Verify history lengths
    assert len(payload["cate_history"]) == 50
    assert len(payload["brand_history"]) == 50
    assert len(payload["btag_history"]) == 50

    response = client.post(
        "/predict",
        json=payload,
    )

    # Diagnostic output for 422 errors
    print("\nPREDICTION STATUS:", response.status_code)
    print("PREDICTION RESPONSE:", response.text)

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

    response = client.post(
        "/predict",
        json=payload,
    )

    print("\nINVALID PRICE STATUS:", response.status_code)
    print("INVALID PRICE RESPONSE:", response.text)

    assert response.status_code == 422