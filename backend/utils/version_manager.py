import json
import os
from datetime import datetime

VERSION_REGISTRY = "version_registry.json"


def _path(models_dir):
    return os.path.join(models_dir, VERSION_REGISTRY)


def load(models_dir):
    p = _path(models_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(models_dir, reg):
    p = _path(models_dir)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def get_base(models_dir, scene_id):
    reg = load(models_dir)
    cur = reg.get(scene_id, {}).get("current_version", 1)
    for v in reg.get(scene_id, {}).get("versions", []):
        if v["version"] == cur:
            sub = v.get("models_subdir", "")
            return os.path.join(models_dir, sub) if sub else models_dir
    return models_dir


def next_version(models_dir, scene_id):
    reg = load(models_dir)
    vers = reg.get(scene_id, {}).get("versions", [])
    return max((v["version"] for v in vers), default=0) + 1


def register(models_dir, scene_id, version, model_metrics, dataset="", rows=0):
    reg = load(models_dir)
    reg.setdefault(scene_id, {"current_version": 1, "versions": []})
    sub = "" if version == 1 else f"v{version}"
    entry = {
        "version": version,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset, "rows": rows,
        "models_subdir": sub, "models": model_metrics,
    }
    reg[scene_id]["versions"] = [
        v for v in reg[scene_id]["versions"] if v["version"] != version
    ]
    reg[scene_id]["versions"].append(entry)
    reg[scene_id]["versions"].sort(key=lambda x: x["version"])
    reg[scene_id]["current_version"] = version
    save(models_dir, reg)
    return entry


def activate(models_dir, scene_id, version):
    reg = load(models_dir)
    if not any(v["version"] == version for v in reg.get(scene_id, {}).get("versions", [])):
        return False, f"Version {version} not found"
    reg[scene_id]["current_version"] = version
    save(models_dir, reg)
    return True, f"Switched to v{version}"


def list_versions(models_dir, scene_id):
    reg = load(models_dir)
    cur = reg.get(scene_id, {}).get("current_version", 1)
    vers = reg.get(scene_id, {}).get("versions", [])
    for v in vers:
        v["is_current"] = v["version"] == cur
    return vers
