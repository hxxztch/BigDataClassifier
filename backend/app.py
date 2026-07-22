from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from werkzeug.utils import secure_filename
import time

from database import init_db, create_task_entry, update_task_result, update_task_progress, get_history, get_task_status
from spark_utils import SparkClassifier
from utils.logger import get_logger
from utils.config import PROJECT_ROOT
from utils.config import UPLOAD_DIR
from admin_routes import admin_bp

logger = get_logger(__name__)

app = Flask(__name__)
CORS(app)
app.register_blueprint(admin_bp)

# Upload config
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10GB

# Init
init_db()
classifier = SparkClassifier()

@app.route('/debug/models', methods=['GET'])
def debug_models():
    import os, json
    from utils.config import MODELS_DIR
    models = {
        "models_dir": MODELS_DIR,
        "models_dir_exists": os.path.isdir(MODELS_DIR),
    }
    for m in ["trans_satisfaction_xgboost", "shop_shipping_xgboost", "trans_delay_xgboost"]:
        p = os.path.join(MODELS_DIR, m + '.model')
        models[m] = os.path.isdir(p)
    return jsonify(models)

# Thread pool (max 4 concurrent tasks)
executor = ThreadPoolExecutor(max_workers=4)


def run_async_prediction(task_id, file_path, model_type, scene_type):
    logger.info(f"[Task {task_id}] Backend processing started...")
    try:
        result = classifier.predict(file_path, model_type, scene_type, task_id)
        if "error" in result:
            logger.error(f"[Task {task_id}] Prediction failed: {result['error']}")
            update_task_result(task_id, 0.0, result, status="failed")
        else:
            acc = result.get("accuracy", 0.0)
            update_task_result(task_id, acc, result, status="completed")
            logger.info(f"[Task {task_id}] Completed, acc={acc:.4f}")
    except Exception as e:
        logger.exception(f"[Task {task_id}] System error: {e}")
        update_task_result(task_id, 0.0, {"error": str(e)}, status="failed")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        save_name = f"{int(time.time())}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
        file.save(file_path)
        logger.info(f"File uploaded: {save_name}")
        return jsonify({
            "message": "File uploaded successfully",
            "file_path": file_path,
            "file_name": filename,
        })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    file_path = data.get("file_path")
    model_type = data.get("model_type", "auto")
    scene_type = data.get("scene_type", "unknown")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400
    task_id = create_task_entry(file_path, model_type, scene_type)
    executor.submit(run_async_prediction, task_id, file_path, model_type, scene_type)
    logger.info(f"Task {task_id} submitted (pool)")
    return jsonify({"task_id": task_id, "status": "processing"})


@app.route("/task/<int:task_id>", methods=["GET"])
def get_task(task_id):
    task = get_task_status(task_id)
    if task:
        response = {
            "task_id": task["task_id"],
            "status": task["status"],
            "result": task.get("result"),
            "finish_time": task["finish_time"],
            "current_model": task["current_model"],
            "model_progress_status": task["model_progress_status"],
        }
        return jsonify(response)
    else:
        return jsonify({"error": "Task not found"}), 404


import os as _os
_STATIC_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "static")

@app.route("/static/<path:filename>")
def serve_static(filename):
    from flask import send_from_directory
    return send_from_directory(_STATIC_DIR, filename)


@app.route("/", methods=["GET"])
def index():
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    html_path = os.path.join(frontend_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    resp = app.make_response(html)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/history", methods=["GET"])
def history():
    scene_type = request.args.get("scene_type")
    tasks = get_history(scene_type)
    return jsonify(tasks)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
