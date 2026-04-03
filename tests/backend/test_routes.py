from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app

client = TestClient(app)

MOCK_RESULT = {
    "verdict": "fake",
    "fake_probability": 87.3,
    "real_probability": 12.7,
    "confidence": 87.3,
    "face_detected": True,
    "face_crop": "base64encodedstring",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


@patch("app.core.service.analyze_image", new_callable=AsyncMock, return_value=MOCK_RESULT)
def test_analyze_endpoint(mock_analyze):
    r = client.post(
        "/api/v1/analyze",
        files={"file": ("test.jpg", b"fake", "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["verdict"] == "fake"
    assert data["confidence"] == 87.3
