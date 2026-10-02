"""
AI Based Road Damage Severity Mapping  -  Flask web application
Run:  python app.py     then open  http://127.0.0.1:5000
"""
import os
import csv
import io
import uuid
from flask import (Flask, render_template, request, redirect, url_for, flash,
                   jsonify, Response, abort)

import config
import database
import detector

BASE_DIR=
os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,template_folder=os.path.join(BASE_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.secret_key = "road-damage-demo-key"

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.RESULT_FOLDER, exist_ok=True)
database.init_db()


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.context_processor
def inject_globals():
    return {"db_backend": database.backend(), "model_name": "YOLO model" if detector.yolo_available() else "OpenCV detector"}


@app.route("/")
def index():
    recent = database.list_detections(limit=4)
    return render_template("index.html", recent=recent)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please choose a road image first.", "error")
        return redirect(url_for("index"))
    if not allowed(file.filename):
        flash("Only JPG, PNG, BMP or WEBP images are allowed.", "error")
        return redirect(url_for("index"))

    ext = file.filename.rsplit(".", 1)[1].lower()
    name = f"{uuid.uuid4().hex[:12]}.{ext}"
    up_path = os.path.join(config.UPLOAD_FOLDER, name)
    res_name = f"result_{name.rsplit('.', 1)[0]}.jpg"
    res_path = os.path.join(config.RESULT_FOLDER, res_name)
    file.save(up_path)

    try:
        result = detector.analyze_image(up_path, res_path, has_sky=bool(request.form.get("has_sky")))
    except Exception as e:
        os.remove(up_path)
        flash(f"Could not analyse image: {e}", "error")
        return redirect(url_for("index"))

    location = request.form.get("location", "").strip()[:200]
    det_id = database.add_detection(name, res_name, location, result)
    return redirect(url_for("result", det_id=det_id))


@app.route("/result/<int:det_id>")
def result(det_id):
    det = database.get_detection(det_id)
    if not det:
        abort(404)
    return render_template("result.html", d=det)


@app.route("/dashboard")
def dashboard():
    sev = request.args.get("severity", "")
    return render_template("dashboard.html", rows=database.list_detections(sev or None),
                           stats=database.stats(), selected=sev)


@app.route("/delete/<int:det_id>", methods=["POST"])
def delete(det_id):
    det = database.get_detection(det_id)
    if det:
        for folder, fname in ((config.UPLOAD_FOLDER, det["original_image"]),
                              (config.RESULT_FOLDER, det["result_image"])):
            try:
                os.remove(os.path.join(folder, fname))
            except OSError:
                pass
        database.delete_detection(det_id)
        flash("Record deleted.", "ok")
    return redirect(url_for("dashboard"))


@app.route("/api/stats")
def api_stats():
    return jsonify(database.stats())


@app.route("/export.csv")
def export_csv():
    rows = database.list_detections(limit=100000)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ID", "Date", "Location", "Damage Type", "Potholes", "Cracks",
                "Damage %", "Severity", "Method", "Image"])
    for r in rows:
        w.writerow([r["id"], r["created_at"], r["location"], r["damage_type"], r["pothole_count"],
                    r["crack_count"], r["damage_percent"], r["severity"], r["method"], r["original_image"]])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=road_damage_report.csv"})


@app.errorhandler(413)
def too_large(_):
    flash("Image too large (max 10 MB).", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
