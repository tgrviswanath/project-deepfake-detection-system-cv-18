from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import io
import torch
from app.main import app

client = TestClient(app)


def _sample_image() -> bytes:
    img = Image.new("RGB", (300, 300), color=(200, 180, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _mock_model():
    model = MagicMock()
    model.return_value = torch.tensor([[0.8]])
    return model


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@patch("app.core.detector._get_model", return_value=_mock_model())
@patch("app.core.detector._crop_face", return_value=None)
def test_analyze_fake(mock_crop, mock_model):
    r = client.post(
        "/api/v1/cv/analyze",
        files={"file": ("test.jpg", _sample_image(), "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "verdict" in data
    assert "fake_probability" in data
    assert "real_probability" in data
    assert "face_crop" in data


def test_analyze_unsupported_format():
    r = client.post(
        "/api/v1/cv/analyze",
        files={"file": ("test.gif", b"GIF89a", "image/gif")},
    )
    assert r.status_code == 400


def test_analyze_empty_file():
    r = client.post(
        "/api/v1/cv/analyze",
        files={"file": ("test.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400
