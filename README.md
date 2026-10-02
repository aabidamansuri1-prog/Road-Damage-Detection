# AI Based Road Damage Severity Mapping

Web app: upload a road image -> detect potholes & cracks -> classify severity (Low / Medium / High)
-> save in database -> view on dashboard.

**Stack (same as synopsis):** HTML/CSS/JS + Python Flask + OpenCV + MySQL.

## 1. Run it (5 minutes)

```bash
# 1. open a terminal inside this folder, then (recommended) make a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows      (Mac/Linux: source venv/bin/activate)

# 2. install libraries
pip install -r requirements.txt

# 3. create demo images (optional but useful)
python make_samples.py

# 4. start the app
python app.py
```
Open **http://127.0.0.1:5000** in the browser. Upload images from the `samples/` folder
(or your own road photos).

## 2. MySQL setup

1. Start MySQL (XAMPP / MySQL Workbench / service).
2. Open `config.py` and set `MYSQL_USER` and `MYSQL_PASSWORD` (default user `root`, empty password).
3. Start the app. It creates the database `road_damage_db` and table `detections` automatically
   (`schema.sql` shows the same table if you want to create it manually).
4. The terminal prints `Connected to MySQL database 'road_damage_db'` when it works.

If MySQL is not running the app falls back to a local SQLite file so the demo never crashes
(the footer of every page shows which database is active).

## 3. How it works (for viva)

| Step | What happens | File |
|------|--------------|------|
| 1 | User uploads image (Flask route `/analyze`) | `app.py` |
| 2 | Image resized, converted to grayscale, contrast improved (CLAHE) | `detector.py` |
| 3 | Road background is estimated (heavy Gaussian blur) and subtracted -> damage = darker than surroundings | `detector.py` |
| 4 | Thresholding + morphology (opening/closing) clean the mask | `detector.py` |
| 5 | Contour analysis: big compact blobs = **Pothole**; dark lines (multi-scale Black-Hat filter, catches thin and wide cracks) = **Crack** | `detector.py` |
| 5b | *Crack density* = share of the road (8x8 grid) containing cracks. Cracks everywhere = "alligator cracking" = High | `crack_density()` |
| 6 | Severity calculated from damaged-area % and pothole size/count | `classify_severity()` |
| 7 | Result saved in MySQL table `detections` | `database.py` |
| 8 | Dashboard shows statistics, severity chart, records, CSV export | `templates/dashboard.html` |

**Severity rules**

| Severity | Condition |
|----------|-----------|
| High   | damaged area >= 10 %  OR  one pothole >= 6 % of road area  OR  4+ potholes  OR  crack density >= 55 % |
| Medium | damaged area >= 3 %   OR  one pothole >= 2 %  OR  2+ potholes  OR  crack density >= 25 % |
| Low    | any smaller damage |
| None   | no damage found |

## 4. Using a trained AI model (optional upgrade)

The default detector is classical computer vision (OpenCV) - fast, no training needed.
To use a deep-learning model instead:
1. `pip install ultralytics`
2. Put a YOLOv8 weights file trained on a road-damage dataset (e.g. RDD2022 or a pothole dataset)
   at `model/best.pt`.
3. Restart the app - it switches to YOLO automatically (footer shows "YOLO model").
   If YOLO fails for any reason it falls back to OpenCV.

## 5. Tips for a good demo
* Use **clear, top-down or slightly angled photos** of tarmac with the damage visible.
* Tick *"Photo includes sky/horizon"* if the photo shows sky or buildings at the top.
* Dark shadows, wet roads and dark vehicles can confuse a classical detector - choose clean images for the live demo.
* Upload 5-6 images before the demo so the dashboard has data.

## 6. Project structure
```
app.py            Flask routes
detector.py       damage detection + severity
database.py       MySQL (with SQLite fallback)
config.py         settings
make_samples.py   creates test images
schema.sql        MySQL table
templates/        HTML pages     static/  CSS, uploaded & result images
```
