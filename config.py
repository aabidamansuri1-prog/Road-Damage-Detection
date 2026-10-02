"""Project configuration. Change the MySQL values below (or set environment variables)."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

# ---- Database -------------------------------------------------------------
# DB_TYPE = "mysql"  -> uses MySQL (as in the synopsis)
# DB_TYPE = "sqlite" -> uses a local file, no setup needed
# If MySQL is selected but not reachable, the app automatically falls back to SQLite
# so your demo never crashes.
DB_TYPE = os.environ.get("DB_TYPE", "mysql")

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", 3306))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DB = os.environ.get("MYSQL_DB", "road_damage_db")

SQLITE_PATH = os.path.join(BASE_DIR, "road_damage.db")

# ---- Optional trained model ----------------------------------------------
# Put a YOLOv8 weights file here (trained on a road damage dataset) to use it
# instead of the built-in OpenCV detector.
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
