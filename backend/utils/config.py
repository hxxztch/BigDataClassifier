import os, sys, json, subprocess, urllib.parse

if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["SPARK_JAVA_HOME"] = r"C:\Program Files\Java\jdk-21"

from pyspark.sql import SparkSession

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(_BACKEND_DIR, "datasets")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
WAREHOUSE_DIR = os.path.join(PROJECT_ROOT, "spark-warehouse")

# ── 从 scenes.yaml 加载场景配置 ─────────────────────────────
_SCENES_YAML = os.path.join(_BACKEND_DIR, "scenes.yaml")
_SCENE_DATA = {}
if os.path.isfile(_SCENES_YAML):
    try:
        import yaml
        with open(_SCENES_YAML, "r", encoding="utf-8") as f:
            _SCENE_DATA = yaml.safe_load(f)
    except Exception as e:
        print(f"[config] WARNING: scenes.yaml load failed: {e}", file=sys.stderr)

_ALL_SCENES = _SCENE_DATA.get("scenes", {}) if _SCENE_DATA else {}

# 向后兼容: 以下变量由 scenes.yaml 自动生成
DATASET_META      = {k: v["target_col"] for k, v in _ALL_SCENES.items() if v.get("target_col")}
SCENE_LABEL_MAP   = {
    k: {int(lk): lv for lk, lv in (v.get("label_map") or {}).items()}
    for k, v in _ALL_SCENES.items()
}
LABEL_MAP_KEYS    = {scene: sorted(set(m.keys())) for scene, m in SCENE_LABEL_MAP.items()}
SCENE_REQUIRED_COLS = {k: v.get("required_cols") or [] for k, v in _ALL_SCENES.items()}
SCENE_CATEGORIES  = {k: v.get("category", "") for k, v in _ALL_SCENES.items()}
SCENE_NAMES_CN    = {k: v.get("name", k) for k, v in _ALL_SCENES.items()}
SCENE_DESCRIPTIONS = {k: v.get("description", "") for k, v in _ALL_SCENES.items()}

# ── GPU / RAPIDS 检测 ───────────────────────────────────────
_HAS_GPU = os.environ.get("GPU_ACCELERATION_ENABLED", "").lower() == "true" or (
    subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode == 0
)
_RAPIDS_JAR = os.path.join(PROJECT_ROOT, "jars", "rapids-4-spark_2.12-25.10.0.jar")
_HAS_RAPIDS = os.path.isfile(_RAPIDS_JAR)
_RAPIDS_JAR_URI = "file:///" + urllib.parse.quote(_RAPIDS_JAR.replace("\\", "/"), safe="/:")

# NCCL detection for XGBoost GPU support
_NCCL_AVAILABLE = False
try:
    import importlib.util
    _spec = importlib.util.find_spec("xgboost.collective")
    _NCCL_AVAILABLE = _spec is not None and _HAS_GPU
except Exception:
    pass

# GPU acceleration toggle (persisted via env var, default: auto-detect)
_GPU_USER_DISABLED = os.environ.get("GPU_ACCELERATION_DISABLED", "").lower() == "true"
_GPU_ACCELERATION_ENABLED = _HAS_GPU and (_NCCL_AVAILABLE or _HAS_RAPIDS) and not _GPU_USER_DISABLED


def get_gpu_info():
    info = {
        "has_gpu": _HAS_GPU,
        "has_nccl": _NCCL_AVAILABLE,
        "has_rapids": _HAS_RAPIDS,
        "acceleration_enabled": _GPU_ACCELERATION_ENABLED,
        "user_disabled": _GPU_USER_DISABLED,
        "gpu_name": "",
        "gpu_memory_mb": 0,
        "driver_version": "",
    }
    if _HAS_GPU:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.strip().split(", ")
                info["gpu_name"] = parts[0] if len(parts) > 0 else ""
                info["gpu_memory_mb"] = int(parts[1].replace(" MiB", "")) if len(parts) > 1 else 0
                info["driver_version"] = parts[2] if len(parts) > 2 else ""
        except Exception:
            pass
    info["gpu_name"] = info["gpu_name"] or ("GPU detected" if _HAS_GPU else "No GPU")
    info["usage_note"] = (
        "GPU acceleration requires NVIDIA GPU + NCCL (XGBoost) or RAPIDS jar."
    )
    _GPU_INFO_CACHE = info
    return info


def set_gpu_acceleration(enable: bool):
    os.environ["GPU_ACCELERATION_DISABLED"] = "false" if enable else "true"
    os.environ["DISABLE_RAPIDS"] = "false" if enable else "true"
    global _GPU_USER_DISABLED, _GPU_ACCELERATION_ENABLED
    _GPU_USER_DISABLED = not enable
    _GPU_ACCELERATION_ENABLED = _HAS_GPU and (_NCCL_AVAILABLE or _HAS_RAPIDS) and not _GPU_USER_DISABLED
    return get_gpu_info()


def get_xgboost_device() -> str:
    return "cpu"


def get_project_dirs():
    return {
        "project_root": PROJECT_ROOT,
        "backend": _BACKEND_DIR,
        "models": MODELS_DIR,
        "data": DATA_DIR,
        "uploads": UPLOAD_DIR,
        "warehouse": WAREHOUSE_DIR,
    }


# ── JDK 自动检测 ─────────────────────────────────────────────
def _find_java_home():
    env_home = os.environ.get("SPARK_JAVA_HOME") or os.environ.get("JAVA_HOME")
    if env_home and os.path.isfile(os.path.join(env_home, "bin", "java.exe")):
        return env_home
    import glob
    for pat in [r"C:\Program Files\Java\jdk-*", r"C:\Program Files\Java\jdk1.8*"]:
        for d in sorted(glob.glob(pat), key=str, reverse=True):
            if os.path.isfile(os.path.join(d, "bin", "java.exe")):
                return d
    return None

_SPARK_JAVA_HOME = _find_java_home()
if _SPARK_JAVA_HOME:
    os.environ["JAVA_HOME"] = _SPARK_JAVA_HOME


def get_spark_builder(app_name="SparkPredictor", driver_memory="4g", executor_memory="4g"):
    builder = SparkSession.builder \
        .appName(app_name) \
        .master(os.environ.get("SPARK_MASTER_URL", "local[*]")) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.sql.warehouse.dir", WAREHOUSE_DIR) \
        .config("spark.ui.enabled", "false") \
        .config("spark.rpc.message.maxSize", "256")
    if _HAS_RAPIDS and _GPU_ACCELERATION_ENABLED:
        builder = builder.config("spark.jars", _RAPIDS_JAR_URI)
    return builder


# ── 特殊场景常量 (ind_quality 列数太多, 保持硬编码) ──
IND_QUALITY_DROP_COLS = {"157", "158", "220", "245", "292", "293", "358", "492", "517", "85", "Time"}
IND_QUALITY_FEATURES = [str(i) for i in range(590) if str(i) not in IND_QUALITY_DROP_COLS]

# ── 模型训练相关 ─────────────────────────────────────────────
AUTO_CANDIDATES = ["random_forest", "gbdt", "xgboost", "naive_bayes"]

CLASSIFIER_DISPLAY = {
    "xgboost": "极致梯度提升 (XGBoost)",
    "random_forest": "随机森林 (RF)",
    "gbdt": "梯度提升树 (GBDT)",
    "naive_bayes": "朴素贝叶斯 (NB)",
}
SIMPLE_CLASSIFIERS = {"naive_bayes"}
