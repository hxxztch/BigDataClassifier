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
from utils.config import PROJECT_ROOT, set_shared_spark
from utils.config import UPLOAD_DIR
from admin_routes import admin_bp
from utils.preprocessing import clean_column_names, custom_preprocessing
from utils.data_quality import analyze_dataframe, compare_with_schema

logger = get_logger(__name__)

app = Flask(__name__)
CORS(app)
app.register_blueprint(admin_bp)

@app.route("/api/data-quality", methods=["POST"])
def data_quality():
    data = request.json
    file_path = data.get("file_path")
    scene_type = data.get("scene_type")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400
    try:
        df = classifier.spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
        df = clean_column_names(df)
        df = df.fillna(0).fillna("Unknown")
        df = custom_preprocessing(df, scene_type)
        schema_check = compare_with_schema(df, scene_type)
        dq_report = analyze_dataframe(df, scene_type)
        if "error" in dq_report:
            return jsonify({"error": dq_report["error"]}), 400
        cols_list = list(dq_report["columns"].values())
        # Cache preprocessed data to speed up subsequent prediction
        import hashlib
        cache_key = hashlib.md5((file_path + scene_type).encode()).hexdigest()
        cache_dir = os.path.join(PROJECT_ROOT, "data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_key)  # Delta table (dir)
        from utils.delta_utils import DeltaManager
        _dm = DeltaManager(classifier.spark)
        _dm.write(df, cache_path, mode="overwrite")
        # Log data version history for time travel
        _history = _dm.history(cache_path, limit=3)
        if _history:
            logger.info(f"[Delta] Data cache: {cache_key} ({len(_history)} versions)")
        return jsonify({
            "schema_check": schema_check,
            "total_rows": dq_report["total_rows"],
            "total_columns": dq_report["total_columns"],
            "columns": cols_list,
            "warnings": dq_report.get("warnings", []),
            "cache_key": cache_key,
        })
    except Exception as e:
        logger.exception(f"Data quality check failed: {e}")
        return jsonify({"error": str(e)}), 500

# Upload config
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10GB

# Init
init_db()
from utils.mlflow_utils import init_mlflow
init_mlflow()
classifier = SparkClassifier()
set_shared_spark(classifier.spark)

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
    page = request.args.get("page", 1, type=int)
    page_size = request.args.get("page_size", 50, type=int)
    tasks = get_history(scene_type)
    total = len(tasks)
    start = (page - 1) * page_size
    paged = tasks[start:start + page_size]
    return jsonify({"tasks": paged, "total": total, "page": page, "page_size": page_size})

@app.route('/api/predict/categories', methods=['GET'])
def predict_categories():
    import yaml, os
    sy = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenes.yaml')
    if os.path.isfile(sy):
        with open(sy, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return jsonify(data.get('categories', []))
    return jsonify([])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)

