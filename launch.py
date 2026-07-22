#!/usr/bin/env python3
"""One-click launcher for Spark Big Data Classifier"""
import os, sys, subprocess, webbrowser, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")


def check_java():
    java_candidates = ["java"]
    jh = os.environ.get("JAVA_HOME") or os.environ.get("SPARK_JAVA_HOME")
    if jh:
        java_candidates.insert(0, os.path.join(jh, "bin", "java.exe"))
    for cmd in java_candidates:
        try:
            r = subprocess.run([cmd, "-version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                ver = (r.stderr or r.stdout).split("\n")[0]
                print(f"  Java: {ver}")
                return True
        except Exception:
            pass
    print("  WARNING: Java not found. Install JDK 11/17/21.")
    return False


def check_deps():
    missing = []
    for mod in ["pyspark", "xgboost", "flask", "flask_cors"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"  Installing: {', '.join(missing)} ...")
        req = os.path.join(PROJECT_ROOT, "requirements.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req], check=True)
    else:
        print("  Dependencies: OK")


def train_if_needed(force=False):
    models_dir = os.path.join(PROJECT_ROOT, "models")
    existing = []
    if os.path.isdir(models_dir):
        existing = [d for d in os.listdir(models_dir) if d.endswith(".model")]
    if force or not existing:
        msg = "Retraining..." if force else "Training models (first run)..."
        print(f"  {msg}")
        subprocess.run([sys.executable, "train_models.py"], cwd=BACKEND_DIR, check=True)
    else:
        print(f"  Models: {len(existing)} found")


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 50)
    print("  Spark Big Data Classifier")
    print("=" * 50)
    print()
    print("[1/3] Checking environment...")
    check_java()
    check_deps()
    print()
    print("[2/3] Preparing models...")
    train_if_needed(force="--retrain" in sys.argv)
    print()
    print("[3/3] Starting server...")
    webbrowser.open("http://localhost:5000")
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=BACKEND_DIR)
    print("  Server: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        print("  Server stopped.")


if __name__ == "__main__":
    main()
