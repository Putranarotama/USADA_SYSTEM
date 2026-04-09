# app.py — Flask Web App
# python app.py  →  http://localhost:5000

import os, sys, uuid, json
from flask import (Flask, render_template, request,
                   jsonify, url_for)
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def ok_ext(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400
    f = request.files["file"]
    if not f.filename or not ok_ext(f.filename):
        return jsonify({"error": "Format tidak didukung (JPG/PNG/WEBP)"}), 400

    ext  = f.filename.rsplit('.',1)[1].lower()
    name = f"{uuid.uuid4().hex[:12]}.{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)

    try:
        from inference import get_predictor
        result = get_predictor().predict(path, with_xai=True)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Gagal: {e}"}), 500

    result["image_url"] = url_for("static", filename=f"uploads/{name}")
    if result.get("xai_filename"):
        result["xai_url"] = url_for(
            "static", filename=f"results/{result['xai_filename']}")

    return jsonify(result)


@app.route("/api/species")
def species_api():
    return jsonify([
        {"id": k, "name": k.replace("_"," "),
         "latin": v.get("latin","-"),
         "khasiat": v.get("khasiat","-")}
        for k, v in SPECIES_INFO.items()
    ])


@app.route("/api/status")
def status():
    ready  = os.path.exists(BEST_MODEL_PATH)
    n_cls  = 0
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
            n_cls = len(json.load(fh))
    return jsonify({"model_ready": ready, "n_species": n_cls})


if __name__ == "__main__":
    print(f"\n{'='*52}")
    print(f"  Usada Detect — Tanaman Obat Bali")
    print(f"  Buka: http://localhost:{FLASK_PORT}")
    print(f"{'='*52}\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
