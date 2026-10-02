"""
make_samples.py - creates synthetic road images in /samples so you can test the app
immediately (no dataset needed).  Run:  python make_samples.py

NOTE: these are computer-generated, only for testing. For your demo, ALSO try real
road photos (take some pictures of roads with your phone, or download a few from
the public RDD2022 / Kaggle pothole datasets).
"""
import os
import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
W, H = 800, 600


def asphalt():
    base = np.full((H, W), 118, np.float32)
    fine = rng.normal(0, 14, (H, W)).astype(np.float32)
    coarse = cv2.GaussianBlur(rng.normal(0, 40, (H, W)).astype(np.float32), (0, 0), 25)
    grad = np.tile(np.linspace(-12, 12, W, dtype=np.float32), (H, 1))
    g = np.clip(base + fine + coarse + grad, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def pothole(img, cx, cy, rx, ry):
    n = 14
    pts = []
    for i in range(n):
        a = 2 * np.pi * i / n
        r = rng.uniform(0.8, 1.15)
        pts.append([cx + rx * r * np.cos(a), cy + ry * r * np.sin(a)])
    pts = np.array(pts, np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask_s = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255
    dark = np.full_like(img, 35, np.float32) + rng.normal(0, 8, img.shape).astype(np.float32)
    out = img.astype(np.float32) * (1 - mask_s[..., None]) + dark * mask_s[..., None]
    rim = cv2.GaussianBlur(cv2.dilate(mask, np.ones((9, 9), np.uint8)) - mask, (0, 0), 2)
    out += (rim.astype(np.float32) / 255 * 18)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def crack(img, x, y, angle, length, branch=True):
    pts = [(x, y)]
    for _ in range(int(length / 8)):
        angle += rng.normal(0, 0.35)
        x += 8 * np.cos(angle); y += 8 * np.sin(angle)
        pts.append((x, y))
    pts_i = np.array(pts, np.int32).reshape(-1, 1, 2)
    layer = np.zeros((H, W), np.uint8)
    cv2.polylines(layer, [pts_i], False, 255, 2, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8).astype(np.float32) / 255
    out = img.astype(np.float32) * (1 - 0.8 * layer[..., None]) + 20 * 0.8 * layer[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    if branch:
        for k in (len(pts) // 3, 2 * len(pts) // 3):
            bx, by = pts[k]
            out = crack(out, bx, by, angle + rng.choice([-1, 1]) * 0.9, length * 0.4, False)
    return out


def save(name, img):
    cv2.imwrite(os.path.join(OUT, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("created", name)


# 1. clean road
save("sample_1_clean_road.jpg", asphalt())

# 2. single crack  -> Low
img = asphalt()
img = crack(img, 80, 120, 0.5, 520)
save("sample_2_single_crack.jpg", img)

# 3. small pothole -> Low/Medium
img = asphalt()
img = pothole(img, 400, 330, 45, 32)
save("sample_3_small_pothole.jpg", img)

# 4. medium pothole + crack -> Medium
img = asphalt()
img = pothole(img, 330, 330, 75, 52)
img = crack(img, 500, 80, 1.9, 380)
save("sample_4_pothole_and_crack.jpg", img)

# 5. severe: big pothole + many cracks + more potholes -> High
img = asphalt()
img = pothole(img, 400, 300, 150, 100)
img = pothole(img, 140, 450, 60, 45)
img = pothole(img, 660, 130, 55, 40)
img = pothole(img, 640, 470, 50, 38)
img = crack(img, 40, 60, 0.4, 480)
img = crack(img, 760, 300, 3.3, 420)
save("sample_5_severe_damage.jpg", img)
