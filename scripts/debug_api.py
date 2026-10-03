import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

payload = {
    "adgroup_id": 775579,
    "cate_id": 6261,
    "brand": 270915,
    "price": 0.14666666090488434,
    "cate_history": [6261] * 50,
    "brand_history": [270915] * 50,
    "btag_history": [1] * 50,
}

response = client.post("/predict", json=payload)
print("STATUS CODE:", response.status_code)
print("RESPONSE BODY:", response.json())