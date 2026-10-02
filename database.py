"""
database.py - MySQL storage (as in the synopsis) with automatic SQLite fallback.
If MySQL is not running / wrong password, the app still works using a local SQLite file
and prints a clear message in the terminal.
"""
import sqlite3
import config

_backend = None  # "mysql" or "sqlite"  (decided once, in init_db)


def _mysql_conn(with_db=True):
    import pymysql
    return pymysql.connect(
        host=config.MYSQL_HOST, port=config.MYSQL_PORT,
        user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DB if with_db else None,
        cursorclass=pymysql.cursors.DictCursor, autocommit=True,
        connect_timeout=3,
    )


def _sqlite_conn():
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


MYSQL_TABLE = """
CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    original_image VARCHAR(255) NOT NULL,
    result_image VARCHAR(255) NOT NULL,
    location VARCHAR(255) DEFAULT '',
    damage_type VARCHAR(50) NOT NULL,
    pothole_count INT DEFAULT 0,
    crack_count INT DEFAULT 0,
    damage_percent FLOAT DEFAULT 0,
    severity VARCHAR(10) NOT NULL,
    method VARCHAR(30) DEFAULT 'opencv',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""

SQLITE_TABLE = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_image TEXT NOT NULL,
    result_image TEXT NOT NULL,
    location TEXT DEFAULT '',
    damage_type TEXT NOT NULL,
    pothole_count INTEGER DEFAULT 0,
    crack_count INTEGER DEFAULT 0,
    damage_percent REAL DEFAULT 0,
    severity TEXT NOT NULL,
    method TEXT DEFAULT 'opencv',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""


def init_db():
    """Create database + table. Decides which backend is used."""
    global _backend
    if config.DB_TYPE == "mysql":
        try:
            c = _mysql_conn(with_db=False)
            with c.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DB}`")
            c.close()
            c = _mysql_conn()
            with c.cursor() as cur:
                cur.execute(MYSQL_TABLE)
            c.close()
            _backend = "mysql"
            print(f"[database] Connected to MySQL database '{config.MYSQL_DB}'")
            return _backend
        except Exception as e:
            print(f"[database] MySQL not available ({e.__class__.__name__}: {e})")
            print("[database] -> Falling back to SQLite so the app still works.")
    c = _sqlite_conn()
    c.execute(SQLITE_TABLE)
    c.commit()
    c.close()
    _backend = "sqlite"
    print("[database] Using SQLite file:", config.SQLITE_PATH)
    return _backend


def backend():
    return _backend


def _run(query, params=(), fetch=None):
    """Run a query on whichever backend is active. fetch: None | 'one' | 'all' | 'id'"""
    if _backend == "mysql":
        conn = _mysql_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                if fetch == "id":
                    return cur.lastrowid
        finally:
            conn.close()
    else:
        conn = _sqlite_conn()
        try:
            cur = conn.execute(query.replace("%s", "?"), params)
            conn.commit()
            if fetch == "one":
                r = cur.fetchone()
                return dict(r) if r else None
            if fetch == "all":
                return [dict(r) for r in cur.fetchall()]
            if fetch == "id":
                return cur.lastrowid
        finally:
            conn.close()


def _clean(row):
    """Make a row JSON/template friendly (datetime -> text)."""
    if row and row.get("created_at") is not None:
        row["created_at"] = str(row["created_at"])[:19]
    return row


def add_detection(original, result, location, r):
    return _run(
        """INSERT INTO detections
           (original_image, result_image, location, damage_type, pothole_count,
            crack_count, damage_percent, severity, method)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (original, result, location, r["damage_type"], r["pothole_count"],
         r["crack_count"], r["damage_percent"], r["severity"], r["method"]),
        fetch="id",
    )


def get_detection(det_id):
    return _clean(_run("SELECT * FROM detections WHERE id=%s", (det_id,), fetch="one"))


def list_detections(severity=None, limit=200):
    if severity in ("Low", "Medium", "High", "None"):
        rows = _run("SELECT * FROM detections WHERE severity=%s ORDER BY id DESC LIMIT %s",
                    (severity, limit), fetch="all")
    else:
        rows = _run("SELECT * FROM detections ORDER BY id DESC LIMIT %s", (limit,), fetch="all")
    return [_clean(r) for r in rows]


def delete_detection(det_id):
    _run("DELETE FROM detections WHERE id=%s", (det_id,))


def stats():
    total = _run("SELECT COUNT(*) AS n FROM detections", fetch="one")["n"]
    sev = {"None": 0, "Low": 0, "Medium": 0, "High": 0}
    for r in _run("SELECT severity, COUNT(*) AS n FROM detections GROUP BY severity", fetch="all"):
        sev[r["severity"]] = r["n"]
    agg = _run("""SELECT COALESCE(SUM(pothole_count),0) AS potholes,
                         COALESCE(SUM(crack_count),0) AS cracks,
                         COALESCE(AVG(damage_percent),0) AS avg_damage
                  FROM detections""", fetch="one")
    return {
        "total": int(total),
        "severity": sev,
        "potholes": int(agg["potholes"]),
        "cracks": int(agg["cracks"]),
        "avg_damage": round(float(agg["avg_damage"]), 2),
    }
