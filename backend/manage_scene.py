
#!/usr/bin/env python
import os, sys, csv, shutil, glob
from collections import Counter
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BACKEND_DIR)
_SCENES_YAML = os.path.join(_BACKEND_DIR, "scenes.yaml")
_DATA_DIR = os.path.join(_BACKEND_DIR, "datasets")
_MODELS_DIR = os.path.join(os.path.dirname(_BACKEND_DIR), "models")
import yaml

def _load_yaml():
    if not os.path.isfile(_SCENES_YAML):
        return {"scenes": {}}
    with open(_SCENES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"scenes": {}}

def _save_yaml(data):
    os.makedirs(os.path.dirname(_SCENES_YAML), exist_ok=True)
    with open(_SCENES_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print("[OK] scenes.yaml saved")

def cmd_list(args):
    data = _load_yaml()
    scenes = data.get("scenes", {})
    if not scenes: print("(no scenes)"); return
    print(f'{"Scene":<28} {"Name":<22} {"Category":<12} {"Target":<22} {"Preproc":<10}')
    print("-" * 100)
    for sid, s in sorted(scenes.items()):
        pre = s.get("preprocessing", "") or "-"
        name = s.get("name", "") or sid
        cat = s.get("category", "") or "-"
        tgt = s.get("target_col", "") or "-"
        print(f"{sid:<28} {name:<22} {cat:<12} {tgt:<22} {pre:<10}")

def cmd_show(args):
    scene_id = args[0]
    data = _load_yaml()
    s = data.get("scenes", {}).get(scene_id)
    if not s: print(f"Scene [{scene_id}] not found"); return
    for k in ["name","category","description","target_col","preprocessing"]:
        print(f"  {k}: {s.get(k, '-')}")
    print(f"  label_map: {s.get('label_map', {})}")
    req = s.get("required_cols") or []
    print(f"  required_cols ({len(req)}): {', '.join(req[:5])}..." if req else "  required_cols: 0")

def cmd_add(args):
    csv_path = args[0]
    if not os.path.isfile(csv_path):
        print(f"[ERROR] File not found: {csv_path}"); return
    filename = os.path.basename(csv_path)
    scene_id = os.path.splitext(filename)[0]
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        col_values = {h: Counter() for h in headers}
        for i, row in enumerate(reader):
            if i >= 5000: break
            for h, v in zip(headers, row): col_values[h][v.strip()] += 1
        all_cols = headers
    candidates = [h for h in headers if len(col_values[h]) <= 2]
    target_guess = headers[-1]
    for kw in ["class","label","target","churn","failure","quality","severity","ordered","satisfaction","fraud","del15","y_n"]:
        for c in candidates:
            if kw in c.lower(): target_guess = c; break
        else: continue; break
    print(f"File: {filename}, Cols: {len(all_cols)}, Detected target: [{target_guess}]")
    target_col = input(f"Target (default: {target_guess}): ").strip() or target_guess
    label_map = {}
    for val, cnt in sorted(col_values[target_col].items(), key=lambda x: -x[1]):
        name = input(f"  Value '{val}' ({cnt}) label: ").strip()
        label_map[val] = name or f"Class_{val}"
    scene_name = input(f"Scene name (default: {scene_id}): ").strip() or scene_id
    category = input(f"Category (default: Unclassified): ").strip() or "Unclassified"
    desc = input(f"Description (default: Predict {target_col}): ").strip() or f"Predict {target_col}"
    copy_it = input("Copy to datasets/? (Y/n): ").strip().lower() != "n"
    int_lm = {}
    for k,v in label_map.items():
        try: int_lm[int(k)] = v
        except ValueError: int_lm[k] = v
    new_scene = {"name":scene_name,"category":category,"description":desc,"target_col":target_col,"label_map":int_lm,"required_cols":all_cols[:]}
    data = _load_yaml()
    data.setdefault("scenes", {})[scene_id] = new_scene
    _save_yaml(data)
    if copy_it:
        os.makedirs(_DATA_DIR, exist_ok=True)
        shutil.copy2(csv_path, os.path.join(_DATA_DIR, filename))
        print("[OK] Copied to datasets/")
    print(f"Scene [{scene_id}] added! Train: python manage_scene.py train {scene_id}")

def cmd_remove(args):
    scene_id = args[0]
    data = _load_yaml()
    if scene_id not in data.get("scenes", {}): print(f"Scene [{scene_id}] not found"); return
    if input(f"Remove [{scene_id}]? (y/N): ").strip().lower() == "y":
        del data["scenes"][scene_id]; _save_yaml(data)

def cmd_train(args):
    scene_id = args[0]
    data = _load_yaml()
    if scene_id not in data.get("scenes", {}): print(f"Scene [{scene_id}] not found"); return
    csv_path = os.path.join(_DATA_DIR, f"{scene_id}.csv")
    if not os.path.isfile(csv_path):
        for f in glob.glob(os.path.join(_DATA_DIR, "*.csv")):
            if scene_id in os.path.basename(f): csv_path = f; break
    if not os.path.isfile(csv_path):
        print(f"[ERROR] No CSV for [{scene_id}] in datasets/"); return
    from utils.config import get_spark_builder
    from train_models import train_one_file
    spark = get_spark_builder(app_name=f"Train_{scene_id}", driver_memory="4g").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try: train_one_file(spark, csv_path)
    finally: spark.stop()

def cmd_train_all(args):
    data = _load_yaml()
    if not data.get("scenes"): print("(no scenes)"); return
    from utils.config import get_spark_builder
    spark = get_spark_builder(app_name="TrainAll", driver_memory="4g").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        for scene_id in data["scenes"]:
            csv_path = os.path.join(_DATA_DIR, f"{scene_id}.csv")
            if os.path.isfile(csv_path):
                from train_models import train_one_file
                print(f"\\nTraining: {scene_id}")
                train_one_file(spark, csv_path)
            else:
                print(f"Skip {scene_id}: no CSV")
    finally: spark.stop()

def main():
    cmds = {"list":cmd_list,"show":cmd_show,"add":cmd_add,"remove":cmd_remove,"train":cmd_train,"train-all":cmd_train_all}
    if len(sys.argv)<2 or sys.argv[1] not in cmds:
        print("Commands: list, show <s>, add <csv>, remove <s>, train <s>, train-all"); return
    cmds[sys.argv[1]](sys.argv[2:])

if __name__ == "__main__": main()
