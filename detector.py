"""
detector.py  -  Road damage detection and severity classification.

Two detection back-ends:
  1. OpenCV detector (default, works out of the box, no training needed)
  2. YOLO detector   (used automatically if model/best.pt exists and `ultralytics` is installed)

Both return the same result dictionary, so the rest of the app doesn't care which one ran.
"""
import os
import cv2
import numpy as np

import config

MAX_WIDTH = 900  # images are resized to this width for processing (speed + consistent thresholds)

# BGR colours used on the annotated image
COLORS = {"Low": (80, 200, 80), "Medium": (0, 165, 255), "High": (50, 50, 230)}


# --------------------------------------------------------------------------
# Severity rules (explained in README / viva notes)
# --------------------------------------------------------------------------
def classify_severity(damage_percent, pothole_count, largest_pothole_pct, crack_density=0.0):
    """
    Return 'None', 'Low', 'Medium' or 'High' for the whole image.

    damage_percent       : % of road area covered by detected damage
    pothole_count        : number of potholes found
    largest_pothole_pct  : area of biggest pothole as % of road area
    crack_density        : 0-1, share of the road (split in a grid) that contains cracks.
                           High density = "alligator" / network cracking = serious damage.
    """
    if damage_percent <= 0 and pothole_count == 0:
        return "None"
    # High: lots of damage, or one very big pothole, or many potholes, or cracks everywhere
    if (damage_percent >= 10 or largest_pothole_pct >= 6 or pothole_count >= 4
            or crack_density >= 0.55):
        return "High"
    # Medium: noticeable damage or any decent-sized pothole or widespread cracking
    if (damage_percent >= 3 or largest_pothole_pct >= 2 or pothole_count >= 2
            or crack_density >= 0.25):
        return "Medium"
    return "Low"


def _single_severity(kind, area_pct):
    """Severity of ONE detected region, used only for box colour."""
    if kind == "Pothole":
        return "High" if area_pct >= 6 else "Medium" if area_pct >= 2 else "Low"
    return "High" if area_pct >= 3 else "Medium" if area_pct >= 1 else "Low"


def _damage_label(potholes, cracks):
    if potholes and cracks:
        return "Pothole + Crack"
    if potholes:
        return "Pothole"
    if cracks:
        return "Crack"
    return "None"


def _draw(img, x, y, w, h, kind, severity, conf=None):
    color = COLORS[severity]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    text = f"{kind} ({severity})" if conf is None else f"{kind} {conf:.0%} ({severity})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(y - th - 8, 0)
    cv2.rectangle(img, (x, top), (x + tw + 6, top + th + 8), color, -1)
    cv2.putText(img, text, (x + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def _resize(img):
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        s = MAX_WIDTH / w
        img = cv2.resize(img, (MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------------------------
# 1) OpenCV detector
# --------------------------------------------------------------------------
def _make_roi(H, W, has_sky):
    roi = np.full((H, W), 255, np.uint8)
    if has_sky:
        roi[: int(H * 0.28), :] = 0          # ignore sky / trees / horizon
    b = max(4, int(0.01 * W))
    roi[:b, :] = 0; roi[-b:, :] = 0; roi[:, :b] = 0; roi[:, -b:] = 0
    return roi


def detect_opencv(img, has_sky=False):
    """
    Classical computer-vision pipeline:
      grey -> CLAHE contrast boost -> POTHOLES: dark compact blobs (background subtraction
      + global darkness test) ; CRACKS: thin dark lines found with a multi-scale Black-Hat
      filter (so both thin and wide cracks are caught) -> contour analysis.
    Returns: detections, damage_mask, crack_mask, roi
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)

    roi = _make_roi(H, W, has_sky)
    roi_area = float(np.count_nonzero(roi))

    # ---------------- potholes ----------------
    background = cv2.GaussianBlur(smooth, (0, 0), sigmaX=max(15, W / 25))
    diff = cv2.subtract(background, smooth).astype(np.float32)
    diff[roi == 0] = 0
    vals = diff[roi > 0]
    thr = max(22.0, float(vals.mean() + 2.2 * vals.std())) if vals.size else 25.0
    dark = (diff > thr).astype(np.uint8) * 255

    # very large potholes look "normal" next to their own surroundings, so also flag
    # anything much darker than the typical road grey
    raw = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.float32)
    road_vals = raw[roi > 0]
    if road_vals.size:
        med = float(np.median(road_vals))
        mad = 1.4826 * float(np.median(np.abs(road_vals - med)))
        gthr = max(38.0, 3.5 * mad)
        dark_global = ((med - raw) > gthr) & (roi > 0)
        dark = cv2.bitwise_or(dark, dark_global.astype(np.uint8) * 255)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blobs = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_open)
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, k_close)

    detections = []
    damage_mask = np.zeros((H, W), np.uint8)
    pothole_mask = np.zeros((H, W), np.uint8)

    cnts, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_pothole = 0.005 * roi_area
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_pothole:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c)) or 1
        solidity = area / hull_area
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if solidity < 0.6 or aspect > 4.0:       # ragged / elongated -> shadow, stain or crack
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(damage_mask, [c], -1, 255, -1)
        cv2.drawContours(pothole_mask, [c], -1, 255, -1)
        detections.append(("Pothole", x, y, w, h, 100.0 * area / roi_area))

    # ---------------- cracks (multi-scale) ----------------
    lines = np.zeros((H, W), np.uint8)
    for k in (max(9, W // 55), max(17, W // 26), max(33, W // 13)):
        k |= 1  # odd
        kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bh = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kb)
        bh[roi == 0] = 0
        bv = bh[roi > 0]
        t = max(18.0, float(bv.mean() + 2.5 * bv.std())) if bv.size else 20.0
        lines = cv2.bitwise_or(lines, (bh > t).astype(np.uint8) * 255)

    # long / ragged dark regions that are NOT potholes are crack networks -> count them as cracks
    ragged = cv2.bitwise_and(blobs, cv2.bitwise_not(cv2.dilate(pothole_mask, k_open)))
    lines = cv2.bitwise_or(lines, ragged)
    lines = cv2.bitwise_and(lines, cv2.bitwise_not(cv2.dilate(pothole_mask, k_open)))
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # remove noise: keep only connected pieces that are long enough to be a crack
    n, labels, st, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    crack_mask = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if max(w, h) < 0.05 * W or area < 25:
            continue
        crack_mask[labels == i] = 255
    crack_mask = cv2.dilate(crack_mask, np.ones((3, 3), np.uint8))
    damage_mask = cv2.bitwise_or(damage_mask, crack_mask)

    # merge neighbouring crack pieces into "crack regions" (one box per region)
    merged = cv2.dilate(crack_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) < 0.07 * W:
            continue
        region = np.zeros((H, W), np.uint8)
        cv2.drawContours(region, [c], -1, 255, -1)
        crack_px = float(np.count_nonzero(cv2.bitwise_and(crack_mask, region)))
        detections.append(("Crack", x, y, w, h, 100.0 * crack_px / roi_area * 3))
    return detections, damage_mask, crack_mask, roi


def crack_density(crack_mask, roi, grid=8):
    """Share of the road area (split into a grid) that contains cracks. 0 = none, 1 = everywhere."""
    ys, xs = np.nonzero(roi)
    if ys.size == 0:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cm, rm = crack_mask[y0:y1, x0:x1], roi[y0:y1, x0:x1]
    hs, ws = max(1, (y1 - y0) // grid), max(1, (x1 - x0) // grid)
    cracked = total = 0
    for gy in range(grid):
        for gx in range(grid):
            cell = cm[gy * hs:(gy + 1) * hs, gx * ws:(gx + 1) * ws]
            rcell = rm[gy * hs:(gy + 1) * hs, gx * ws:(gx + 1) * ws]
            if rcell.size == 0 or np.count_nonzero(rcell) < 0.5 * rcell.size:
                continue
            total += 1
            if np.count_nonzero(cell) >= 0.02 * cell.size:
                cracked += 1
    return cracked / total if total else 0.0


# --------------------------------------------------------------------------
# 2) Optional YOLO detector
# --------------------------------------------------------------------------
_yolo = None


def yolo_available():
    return os.path.exists(config.YOLO_MODEL_PATH)


def detect_yolo(img):
    global _yolo
    from ultralytics import YOLO  # imported lazily so the app works without it
    if _yolo is None:
        _yolo = YOLO(config.YOLO_MODEL_PATH)
    H, W = img.shape[:2]
    res = _yolo.predict(img, conf=0.25, verbose=False)[0]
    detections, confs = [], []
    mask = np.zeros((H, W), np.uint8)
    crack_mask = np.zeros((H, W), np.uint8)
    for b in res.boxes:
        name = str(res.names[int(b.cls)]).lower()
        x1, y1, x2, y2 = [max(0, int(v)) for v in b.xyxy[0].tolist()]
        # RDD2022 classes: D00/D10/D20 = cracks, D40 = pothole
        kind = "Pothole" if ("pothole" in name or "d40" in name) else "Crack"
        w, h = x2 - x1, y2 - y1
        mask[y1:y2, x1:x2] = 255
        if kind == "Crack":
            crack_mask[y1:y2, x1:x2] = 255
        detections.append((kind, x1, y1, w, h, 100.0 * w * h / (W * H)))
        confs.append(float(b.conf))
    return detections, mask, crack_mask, np.full((H, W), 255, np.uint8), confs


# --------------------------------------------------------------------------
# Public function used by app.py
# --------------------------------------------------------------------------
def analyze_image(path, result_path, has_sky=False):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read the image. Please upload a valid JPG/PNG file.")
    img = _resize(img)

    confs = []
    method = "opencv"
    result = None
    if yolo_available():
        try:
            result = detect_yolo(img)
            confs = result[4]
            method = "yolo"
        except Exception as e:                        # fall back silently to OpenCV
            print("[detector] YOLO failed, using OpenCV:", e)
            result = None
    if result is None:
        result = detect_opencv(img, has_sky)
    dets, mask, crack_mask, roi = result[:4]
    roi_area = float(np.count_nonzero(roi))

    potholes = [d for d in dets if d[0] == "Pothole"]
    cracks = [d for d in dets if d[0] == "Crack"]
    damage_percent = min(100.0, 100.0 * np.count_nonzero(mask) / roi_area) if roi_area else 0.0
    largest = max([d[5] for d in potholes], default=0.0)
    density = crack_density(crack_mask, roi)
    severity = classify_severity(damage_percent, len(potholes), largest, density)

    # ---- draw result: coloured damage pixels + labelled boxes ----
    out = img.copy()
    pothole_only = cv2.bitwise_and(mask, cv2.bitwise_not(crack_mask))
    tint = out.copy()
    tint[pothole_only > 0] = (0, 140, 255)                       # potholes: orange tint
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)
    out[crack_mask > 0] = (40, 40, 235)                          # cracks: red
    for i, (kind, x, y, w, h, pct) in enumerate(dets):
        _draw(out, x, y, w, h, kind, _single_severity(kind, pct),
              confs[i] if i < len(confs) else None)

    banner = (f"Overall severity: {severity}  |  Damage area: {damage_percent:.1f}%"
              f"  |  Crack density: {density:.0%}")
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(out, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(result_path, out)

    return {
        "damage_type": _damage_label(potholes, cracks),
        "pothole_count": len(potholes),
        "crack_count": len(cracks),
        "damage_percent": round(float(damage_percent), 2),
        "severity": severity,
        "method": method,
    }
