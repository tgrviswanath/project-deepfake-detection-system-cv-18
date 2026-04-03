"""
Deepfake detection using EfficientNet-B0 (PyTorch).
- Crops face region using OpenCV DNN SSD
- Classifies cropped face as real or fake
- Returns verdict, confidence, and face crop as base64
"""
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import requests
import torch
import torch.nn as nn
from torchvision import transforms, models
from app.core.config import settings

_net = None       # OpenCV face detector
_model = None     # EfficientNet classifier

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _download_face_detector():
    os.makedirs("models", exist_ok=True)
    for path, url in [(settings.PROTOTXT_PATH, PROTOTXT_URL), (settings.CAFFEMODEL_PATH, CAFFEMODEL_URL)]:
        if not os.path.exists(path):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)


def _get_face_net():
    global _net
    if _net is None:
        _download_face_detector()
        _net = cv2.dnn.readNetFromCaffe(settings.PROTOTXT_PATH, settings.CAFFEMODEL_PATH)
    return _net


def _get_model():
    global _model
    if _model is None:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        if os.path.exists(settings.MODEL_PATH):
            model.load_state_dict(torch.load(settings.MODEL_PATH, map_location="cpu"))
        model.eval()
        _model = model
    return _model


def _load_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > settings.MAX_IMAGE_SIZE:
        scale = settings.MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    return np.array(img)


def _crop_face(img_rgb: np.ndarray) -> np.ndarray | None:
    net = _get_face_net()
    h, w = img_rgb.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        1.0, (300, 300), (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    dets = net.forward()
    best_conf, best_box = 0.0, None
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf > best_conf:
            best_conf = conf
            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            best_box = (x1, y1, x2, y2)
    if best_conf < 0.3 or best_box is None:
        return None
    x1, y1, x2, y2 = best_box
    return img_rgb[y1:y2, x1:x2]


def _to_base64(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def analyze(image_bytes: bytes) -> dict:
    img = _load_image(image_bytes)
    face = _crop_face(img)
    input_img = Image.fromarray(face if face is not None else img)

    tensor = _transform(input_img).unsqueeze(0)
    model = _get_model()
    with torch.no_grad():
        logit = model(tensor)[0, 0]
        prob_fake = float(torch.sigmoid(logit))

    verdict = "fake" if prob_fake >= settings.CONFIDENCE_THRESHOLD else "real"
    confidence = prob_fake if verdict == "fake" else 1.0 - prob_fake

    return {
        "verdict": verdict,
        "fake_probability": round(prob_fake * 100, 2),
        "real_probability": round((1 - prob_fake) * 100, 2),
        "confidence": round(confidence * 100, 2),
        "face_detected": face is not None,
        "face_crop": _to_base64(face if face is not None else img),
    }
