# -*- coding: utf-8 -*-
"""
Automated end-to-end test: Spark MLlib training + sklearn training + prediction.
Runs on trans_delay dataset (small, 14 columns, ~607k rows).
Usage: python test_training.py [--cluster]
"""

import os, sys, time, json, glob, shutil, argparse

# ── Ensure JDK 17 ──
os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Java\jdk-17.0.20")

# ── Parse args ──
parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true", help="Use standalone cluster mode")
args = parser.parse_args()

if args.cluster:
    os.environ["SPARK_MASTER_URL"] = "spark://127.0.0.1:7077"
    print("[MODE] Cluster mode (spark://127.0.0.1:7077)")
else:
    os.environ["SPARK_MASTER_URL"] = "local[*]"
    print("[MODE] Local mode (local[*])")

# ── Setup paths ──
_BACKEND = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BACKEND)
_DATA_CSV = os.path.join(_BACKEND, "datasets", "trans_delay.csv")
_MODELS_DIR = os.path.join(os.path.dirname(_BACKEND), "models")

if not os.path.isfile(_DATA_CSV):
    print(f"[ERROR] Dataset not found: {_DATA_CSV}")
    sys.exit(1)

# ── Clean old test artifacts ──
for _pat in ["trans_delay_random_forest_mllib", "trans_delay_gbdt_mllib",
             "trans_delay_naive_bayes_mllib", "trans_delay_xgboost_mllib"]:
    for _d in glob.glob(os.path.join(_MODELS_DIR, "**", _pat), recursive=True):
        shutil.rmtree(_d, ignore_errors=True)

_results = {"tests": [], "passed": 0, "failed": 0}

def _ok(name, detail=""):
    _results["tests"].append({"name": name, "status": "PASS", "detail": detail})
    _results["passed"] += 1
    print(f"  [PASS] {name}")

def _fail(name, detail=""):
    _results["tests"].append({"name": name, "status": "FAIL", "detail": detail})
    _results["failed"] += 1
    print(f"  [FAIL] {name}: {detail}")

# ===================================================================
# TEST 1: SparkSession connectivity
# ===================================================================
print("\n" + "="*60)
print("TEST 1: SparkSession connectivity")
print("="*60)

try:
    from utils.config import get_spark_builder
    builder = get_spark_builder("AutoTest_Spark")
    spark = builder.getOrCreate()
    _ver = spark.version
    _master = spark.sparkContext.master
    _java = spark.sparkContext._jvm.System.getProperty("java.version")
    _ok("SparkSession created",
        f"Spark {_ver} | Master {_master} | Java {_java}")

    # Quick sanity
    _cnt = spark.range(10).count()
    if _cnt == 10:
        _ok("DataFrame operations", f"range(10).count() = {_cnt}")
    else:
        _fail("DataFrame operations", f"expected 10, got {_cnt}")
except Exception as e:
    _fail("SparkSession creation", str(e)[:200])
    spark = None

# ===================================================================
# TEST 2: MLlib training (RF / GBT / NB / XGBoost)
# ===================================================================
print("\n" + "="*60)
print("TEST 2: MLlib training on trans_delay.csv")
print("="*60)

if spark is not None:
    try:
        from train_sklearn import train_sklearn

        _t0 = time.time()
        train_sklearn(
            _DATA_CSV,
            scene_id="trans_delay",
            spark=spark,  # reuse existing SparkSession
            progress_callback=lambda p, t: print(f"    [{p}%] {t}")
        )
        _elapsed = time.time() - _t0
        _ok("train_sklearn completed", f"{_elapsed:.1f}s")
    except Exception as e:
        _fail("train_sklearn", str(e)[:300])

# ===================================================================
# TEST 3: MLlib model files verification
# ===================================================================
print("\n" + "="*60)
print("TEST 3: MLlib model files")
print("="*60)

_expected_mllib = ["random_forest", "gbdt", "naive_bayes", "xgboost"]
for _algo in _expected_mllib:
    _found = False
    for _d in glob.glob(os.path.join(_MODELS_DIR, "**", f"trans_delay_{_algo}_mllib"), recursive=True):
        _meta = os.path.join(_d, "metadata.json")
        if os.path.isfile(_meta):
            with open(_meta) as _f:
                _m = json.load(_f)
            _ok(f"MLlib {_algo} model + metadata",
                f"F1={_m['f1_score']} Acc={_m['accuracy']} size={_m['train_size']} backend={_m['backend']}")
            _found = True
            break
    if not _found:
        _fail(f"MLlib {_algo} model", "model directory or metadata.json not found")

# ===================================================================
# TEST 4: Model loading for prediction (_load_model)
# ===================================================================
print("\n" + "="*60)
print("TEST 4: Model loading (_load_model)")
print("="*60)

if spark is not None:
    from spark_utils import SparkClassifier
    _clf = SparkClassifier()
    _clf.spark = spark  # reuse

    for _algo in _expected_mllib:
        _model, _err = _clf._load_model(_algo, "trans_delay")
        if _model is not None:
            _backend = _model.get("backend", "unknown")
            _ok(f"_load_model({_algo})", f"backend={_backend}")
        else:
            _fail(f"_load_model({_algo})", str(_err)[:200])

# ===================================================================
# TEST 5: Prediction pipeline (end-to-end)
# ===================================================================
print("\n" + "="*60)
print("TEST 5: Prediction pipeline")
print("="*60)

if spark is not None:
    try:
        _result = _clf.predict(_DATA_CSV, "auto", "trans_delay", 9999)
        if "error" in _result:
            _fail("Prediction", _result["error"][:200])
        else:
            _ok("Prediction completed",
                f"model={_result['final_model']} accuracy={_result['accuracy']:.4f} "
                f"rows={len(_result['chart_data'])}")
        # Verify key fields
        for _key in ["chart_data", "distribution", "final_model", "confusion_matrix"]:
            if _key in _result:
                _ok(f"  result.{_key}", "present")
            else:
                _fail(f"  result.{_key}", "missing")
    except Exception as e:
        _fail("Prediction pipeline", str(e)[:300])

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Passed: {_results['passed']}  Failed: {_results['failed']}  Total: {len(_results['tests'])}")

if spark is not None:
    spark.stop()

_exit_code = 0 if _results["failed"] == 0 else 1
sys.exit(_exit_code)
