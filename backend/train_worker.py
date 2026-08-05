"""Standalone training worker - runs in subprocess with MKL safeguards."""
import sys, os, json

# Must be set BEFORE any imports that touch numpy or Spark
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["SPARK_MASTER_URL"] = "local[2]"

if len(sys.argv) < 3:
    print("Usage: python train_worker.py <scene_id> <csv_path>")
    sys.exit(1)

scene_id = sys.argv[1]
csv_path = sys.argv[2]
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
status_file = os.path.join(_BACKEND_DIR, "train_status_" + scene_id + ".json")

def _write_status(status, progress=0, error=None, accuracy=None, best_algo=None, progress_text="", results=None):
    d = {"status": status, "progress": progress, "error": error}
    if accuracy is not None: d["accuracy"] = accuracy
    if best_algo is not None: d["best_algo"] = best_algo
    if progress_text: d["progress_text"] = progress_text
    if results is not None: d["results"] = results
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)

_write_status("running", 0, progress_text="?????????...")

from utils.config import PROJECT_ROOT, MODELS_DIR
from train_sklearn import train_sklearn

try:
    def _on_progress(pct, text):
        _write_status("running", pct, progress_text=text)

    _write_status("running", 10, progress_text="正在启动训练进程...")
    train_sklearn(csv_path, scene_id, progress_callback=_on_progress)

    _write_status("running", 90, progress_text="????????...")
    best_acc = 0.0; best_algo = ""; results = []
    algo_names = {"random_forest": "随机森林", "gbdt": "GBDT", "xgboost": "XGBoost", "naive_bayes": "朴素贝叶斯"}
    search_dirs = [MODELS_DIR]
    if os.path.isdir(MODELS_DIR):
        search_dirs += [os.path.join(MODELS_DIR, d) for d in os.listdir(MODELS_DIR)
                        if os.path.isdir(os.path.join(MODELS_DIR, d)) and d.startswith("v")]
    for mt in ["random_forest", "gbdt", "xgboost", "naive_bayes"]:
        found = False
        for sd2 in search_dirs:
            mp = os.path.join(sd2, scene_id + "_" + mt + ".model", "metadata.json")
            if os.path.isfile(mp):
                with open(mp) as mf:
                    _meta = json.load(mf)
                results.append({
                    "algo": algo_names.get(mt, mt),
                    "status": "ok",
                    "accuracy": _meta.get("accuracy", 0),
                    "f1": _meta.get("f1_score", 0)
                })
                if _meta.get("accuracy", 0) > best_acc:
                    best_acc = _meta["accuracy"]
                    best_algo = mt
                found = True
                break
        if not found:
            results.append({"algo": algo_names.get(mt, mt), "status": "fail", "error": "未找到模型"})

    _write_status("completed", 100, accuracy=best_acc, best_algo=best_algo, progress_text="????", results=results)
except Exception as e:
    import traceback
    traceback.print_exc()
    _write_status("failed", error=str(e))
    sys.exit(1)
