"""
Generate sample images for cv-18 Deepfake Detection System.
Run: pip install Pillow numpy && python generate_samples.py
Output: 6 images — 3 "real-looking" faces, 3 "synthetic-looking" faces.
Note: All are synthetic — labels indicate the intended test category.
"""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os

OUT = os.path.dirname(__file__)


def save(img, name):
    img.save(os.path.join(OUT, name))
    print(f"  created: {name}")


def draw_face(d, cx, cy, r, skin, hair, eye_color=(40, 30, 20)):
    d.ellipse([cx - r, cy - r - 10, cx + r, cy + int(r * 0.3)], fill=hair)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=skin)
    for ex in [cx - int(r * 0.3), cx + int(r * 0.3)]:
        er = int(r * 0.13)
        d.ellipse([ex - er - 2, cy - int(r * 0.2) - 2, ex + er + 2, cy - int(r * 0.2) + er * 2 + 2], fill=(255, 255, 255))
        d.ellipse([ex - er, cy - int(r * 0.2), ex + er, cy - int(r * 0.2) + er * 2], fill=eye_color)
    d.polygon([(cx, cy + int(r * 0.1)), (cx - int(r * 0.1), cy + int(r * 0.3)),
               (cx + int(r * 0.1), cy + int(r * 0.3))],
              fill=(int(skin[0] * 0.85), int(skin[1] * 0.85), int(skin[2] * 0.85)))
    d.arc([cx - int(r * 0.3), cy + int(r * 0.35), cx + int(r * 0.3), cy + int(r * 0.55)],
          start=0, end=180, fill=(180, 80, 80), width=max(2, r // 12))


def real_face_1():
    img = Image.new("RGB", (400, 400), (210, 225, 240))
    d = ImageDraw.Draw(img)
    # natural background
    d.rectangle([0, 280, 400, 400], fill=(160, 140, 110))
    draw_face(d, 200, 190, 100, (220, 180, 140), (60, 40, 20))
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    d2 = ImageDraw.Draw(img)
    d2.text((10, 370), "real_sample_1.jpg", fill=(80, 80, 80))
    return img


def real_face_2():
    img = Image.new("RGB", (400, 400), (230, 215, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 300, 400, 400], fill=(80, 120, 60))
    draw_face(d, 200, 185, 95, (180, 130, 90), (20, 15, 10), eye_color=(60, 100, 40))
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    d2 = ImageDraw.Draw(img)
    d2.text((10, 370), "real_sample_2.jpg", fill=(80, 80, 80))
    return img


def real_face_3():
    img = Image.new("RGB", (400, 400), (240, 235, 230))
    d = ImageDraw.Draw(img)
    draw_face(d, 200, 195, 98, (240, 200, 165), (160, 100, 40), eye_color=(80, 60, 40))
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    d2 = ImageDraw.Draw(img)
    d2.text((10, 370), "real_sample_3.jpg", fill=(80, 80, 80))
    return img


def synthetic_face_1():
    """Overly smooth / perfect — typical deepfake artifact simulation."""
    img = Image.new("RGB", (400, 400), (200, 210, 230))
    d = ImageDraw.Draw(img)
    draw_face(d, 200, 190, 100, (225, 185, 148), (55, 38, 18))
    # over-smooth (deepfake-like)
    img = img.filter(ImageFilter.GaussianBlur(2.5))
    # add slight color banding artifact
    arr = np.array(img)
    arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) + 8, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    d2 = ImageDraw.Draw(img)
    d2.text((10, 370), "synthetic_sample_1.jpg", fill=(80, 80, 80))
    return img


def synthetic_face_2():
    """Blurry boundary around face — common deepfake artifact."""
    img = Image.new("RGB", (400, 400), (215, 220, 235))
    d = ImageDraw.Draw(img)
    draw_face(d, 200, 190, 100, (218, 178, 138), (50, 35, 15))
    img = img.filter(ImageFilter.GaussianBlur(3.0))
    arr = np.array(img)
    # add noise in background
    noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    d2 = ImageDraw.Draw(img)
    d2.text((10, 370), "synthetic_sample_2.jpg", fill=(80, 80, 80))
    return img


def synthetic_face_3():
    """Inconsistent lighting — another deepfake tell."""
    img = Image.new("RGB", (400, 400), (205, 215, 225))
    d = ImageDraw.Draw(img)
    draw_face(d, 200, 190, 100, (222, 182, 142), (58, 40, 18))
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    arr = np.array(img)
    # uneven brightness gradient
    for y in range(400):
        factor = 1.0 + 0.15 * (y / 400)
        arr[y] = np.clip(arr[y].astype(float) * factor, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    d2 = ImageDraw.Draw(img)
    d2.text((10, 370), "synthetic_sample_3.jpg", fill=(80, 80, 80))
    return img


if __name__ == "__main__":
    print("Generating cv-18 samples...")
    save(real_face_1(), "real_sample_1.jpg")
    save(real_face_2(), "real_sample_2.jpg")
    save(real_face_3(), "real_sample_3.jpg")
    save(synthetic_face_1(), "synthetic_sample_1.jpg")
    save(synthetic_face_2(), "synthetic_sample_2.jpg")
    save(synthetic_face_3(), "synthetic_sample_3.jpg")
    print("Done — 6 images in samples/")
    print("Tip: real_sample_* → expected 'real', synthetic_sample_* → expected 'fake'")
