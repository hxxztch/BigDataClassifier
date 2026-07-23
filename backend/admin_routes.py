import os, sys, json, threading, csv, shutil, glob
from collections import Counter
from flask import Blueprint, request, jsonify
from utils.version_manager import list_versions as _list_ver, activate as _act_ver, load as _load_reg

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BACKEND_DIR)
_SCENES_YAML = os.path.join(_BACKEND_DIR, "scenes.yaml")
_MODELS_DIR = os.path.join(os.path.dirname(_BACKEND_DIR), "models")
_DATA_DIR = os.path.join(_BACKEND_DIR, "datasets")

import yaml
admin_bp = Blueprint("admin", __name__)

_training_tasks = {}
_lock = threading.Lock()

def _load_yaml():
    if not os.path.isfile(_SCENES_YAML): return {"scenes": {}}
    with open(_SCENES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"scenes": {}}

def _save_yaml(data):
    os.makedirs(os.path.dirname(_SCENES_YAML), exist_ok=True)
    with open(_SCENES_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def _model_status(scene_id):
    mts = ["random_forest", "gbdt", "xgboost", "naive_bayes"]
    result = {}
    for mt in mts:
        path = os.path.join(_MODELS_DIR, f"{scene_id}_{mt}.model")
        mp = os.path.join(path, "metadata.json")
        if os.path.isdir(path) and os.path.isfile(mp):
            with open(mp) as f:
                meta = json.load(f)
            result[mt] = {"exists": True, "accuracy": meta.get("accuracy"),
                          "f1_score": meta.get("f1_score"), "trained_at": meta.get("training_time")}
        else:
            result[mt] = {"exists": False}
    return result

def _run_training(spark, scene_id, csv_path):
    with _lock: _training_tasks[scene_id] = {"status": "running", "progress": 0, "error": None}
    try:
        from train_models import train_one_file
        train_one_file(spark, csv_path)
        with _lock: _training_tasks[scene_id] = {"status": "completed", "progress": 100}
    except Exception as e:
        with _lock: _training_tasks[scene_id] = {"status": "failed", "error": str(e)}

@admin_bp.route("/api/admin/scenes", methods=["GET"])
def list_scenes():
    data = _load_yaml()
    scenes = data.get("scenes", {})
    result = []
    for sid, s in sorted(scenes.items()):
        entry = dict(s); entry["id"] = sid
        entry["models"] = _model_status(sid)
        entry["current_version"] = _load_reg(_MODELS_DIR).get(sid, {}).get("current_version", 1)
        with _lock: entry["training"] = _training_tasks.get(sid, {"status": "idle"})
        result.append(entry)
    return jsonify(result)
@admin_bp.route("/api/admin/scenes/analyze", methods=["POST"])
def analyze_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV"}), 400
    from werkzeug.utils import secure_filename
    os.makedirs(_DATA_DIR, exist_ok=True)
    tmp_path = os.path.join(_DATA_DIR, "_analyze_tmp.csv")
    f.save(tmp_path)
    try:
        # Save CSV to pending dir for add step
        pending_dir = os.path.join(_DATA_DIR, "_pending")
        os.makedirs(pending_dir, exist_ok=True)
        pending_path = os.path.join(pending_dir, f.filename)
        import shutil
        shutil.copy2(tmp_path, pending_path)
        with open(tmp_path, encoding="utf-8-sig") as cf:
            reader = csv.reader(cf)
            headers = next(reader)
            col_values = {h: Counter() for h in headers}
            for i, row in enumerate(reader):
                if i >= 5000: break
                for h, v in zip(headers, row): col_values[h][v.strip()] += 1
        candidates = []
        for h in headers:
            if len(col_values[h]) <= 2: candidates.append(h)
        target_guess = headers[-1]
        for kw in ["class","label","target","churn","failure","quality","severity","ordered","satisfaction","fraud","del15","y_n"]:
            for c in candidates:
                if kw in c.lower(): target_guess = c; break
            else: continue; break
        col_info = []
        for h in headers:
            col_info.append({"name": h, "unique": len(col_values[h]), "is_candidate": h in candidates, "sample_values": list(col_values[h].keys())[:5]})
        return jsonify({"columns": col_info, "total_cols": len(headers), "target_candidates": candidates, "target_guess": target_guess, "filename": f.filename, "pending_path": pending_path})
    finally:
        if os.path.isfile(tmp_path): os.remove(tmp_path)

@admin_bp.route("/api/admin/scenes/add", methods=["POST"])
def add_scene():
    data = request.get_json()
    if not data or "scene_id" not in data or "target_col" not in data:
        return jsonify({"error": "Missing scene_id or target_col"}), 400
    sid = data["scene_id"]
    lm = {}
    for k, v in (data.get("label_map") or {}).items():
        try: lm[int(k)] = v
        except ValueError: lm[k] = v
    new_scene = {
        "name": data.get("name", sid), "category": data.get("category", "Unclassified"),
        "description": data.get("description", ""), "target_col": data["target_col"],
        "label_map": lm, "required_cols": data.get("required_cols", []),
    }
    sd = _load_yaml()
    sd.setdefault("scenes", {})[sid] = new_scene
    _save_yaml(sd)
    # Handle CSV from analyze step or direct upload
    pending = data.get("pending_path")
    if pending and os.path.isfile(pending):
        os.makedirs(_DATA_DIR, exist_ok=True)
        shutil.copy2(pending, os.path.join(_DATA_DIR, f"{sid}.csv"))
    csv_enc = data.get("csv_content")
    if csv_enc and not pending:
        import base64
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(os.path.join(_DATA_DIR, f"{sid}.csv"), "wb") as f:
            f.write(base64.b64decode(csv_enc))
    return jsonify({"ok": True, "scene_id": sid})
@admin_bp.route("/api/admin/scenes/<scene_id>", methods=["DELETE"])
def remove_scene(scene_id):
    sd = _load_yaml()
    if scene_id not in sd.get("scenes", {}):
        return jsonify({"error": "Not found"}), 404
    del sd["scenes"][scene_id]
    _save_yaml(sd)
    return jsonify({"ok": True})

@admin_bp.route("/api/admin/scenes/<scene_id>/train", methods=["POST"])
def train_scene(scene_id):
    sd = _load_yaml()
    if scene_id not in sd.get("scenes", {}):
        return jsonify({"error": "Not found"}), 404
    data = request.get_json() or {}
    csv_path = data.get("dataset_path", "")
    if not csv_path:
        csv_path = os.path.join(_DATA_DIR, f"{scene_id}.csv")
        if not os.path.isfile(csv_path):
            for f in glob.glob(os.path.join(_DATA_DIR, "*.csv")):
                if scene_id in os.path.basename(f): csv_path = f; break
    if not os.path.isfile(csv_path):
        return jsonify({"error": "No CSV"}), 400
    from utils.config import get_spark_builder
    spark = get_spark_builder(app_name=f"AdminTrain_{scene_id}", driver_memory="4g").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    t = threading.Thread(target=_run_training, args=(spark, scene_id, csv_path))
    t.daemon = True; t.start()
    return jsonify({"ok": True, "scene_id": scene_id, "status": "started"})

@admin_bp.route("/api/admin/scenes/train-all", methods=["POST"])
def train_all():
    sd = _load_yaml()
    scenes = sd.get("scenes", {})
    if not scenes: return jsonify({"error": "No scenes"}), 400
    from utils.config import get_spark_builder
    for sid in scenes:
        csv_path = os.path.join(_DATA_DIR, f"{sid}.csv")
        if os.path.isfile(csv_path):
            spark = get_spark_builder(app_name=f"AdminTrain_{sid}", driver_memory="4g").getOrCreate()
            spark.sparkContext.setLogLevel("ERROR")
            t = threading.Thread(target=_run_training, args=(spark, sid, csv_path))
            t.daemon = True; t.start()
    return jsonify({"ok": True})

@admin_bp.route("/api/admin/scenes/training-status", methods=["GET"])
def training_status_all():
    with _lock: return jsonify(dict(_training_tasks))

@admin_bp.route("/api/admin/scenes/<scene_id>/training-status", methods=["GET"])
def training_status_one(scene_id):
    with _lock: s = _training_tasks.get(scene_id, {"status": "idle"})
    return jsonify(s)

@admin_bp.route("/api/admin/versions/<scene_id>", methods=["GET"])
def get_versions(scene_id):
    try:
        vers = _list_ver(_MODELS_DIR, scene_id)
        reg = _load_reg(_MODELS_DIR)
        cur = reg.get(scene_id, {}).get("current_version", 1)
        return jsonify({"versions": vers, "current_version": cur})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@admin_bp.route("/api/admin/versions/<scene_id>/activate", methods=["POST"])
def activate_version(scene_id):
    data = request.get_json()
    version = data.get("version")
    if not version:
        return jsonify({"error": "Missing version"}), 400
    ok, msg = _act_ver(_MODELS_DIR, scene_id, int(version))
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True, "message": msg})
@admin_bp.route("/api/admin/datasets", methods=["GET"])
def list_datasets():
    path = request.args.get("path", "")
    if not path or not os.path.isdir(path):
        path = _DATA_DIR
    files = []
    for f in sorted(os.listdir(path)):
        full = os.path.join(path, f)
        files.append({"name": f, "path": full, "is_dir": os.path.isdir(full), "size": os.path.getsize(full) if os.path.isfile(full) else 0})
    return jsonify({"path": path, "files": files, "parent": os.path.dirname(path)})

@admin_bp.route("/api/admin/datasets/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    target_dir = request.form.get("target_dir", _DATA_DIR)
    os.makedirs(target_dir, exist_ok=True)
    from werkzeug.utils import secure_filename
    save_path = os.path.join(target_dir, secure_filename(f.filename))
    f.save(save_path)
    return jsonify({"ok": True, "path": save_path, "name": f.filename})


