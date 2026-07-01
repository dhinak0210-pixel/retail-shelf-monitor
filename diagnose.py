#!/usr/bin/env python3
"""
diagnose.py - Environment, Database, Model, and Hardware Diagnostic Tool
for Aether-Shelf Retail Shelf Monitor.

Runs a series of sanity checks to verify setup and benchmark performance.
"""

import os
import sys
import time
import platform
import shutil
import sqlite3
import numpy as np
from pathlib import Path

# Terminal Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
ENDC = "\033[0m"

def print_header(title):
    print(f"\n{BOLD}{CYAN}{'='*60}{ENDC}")
    print(f"{BOLD}{CYAN}  {title}{ENDC}")
    print(f"{BOLD}{CYAN}{'='*60}{ENDC}")

def print_check(name, status, details=""):
    if status == "ok":
        print(f"  {GREEN}✓{ENDC} {BOLD}{name}{ENDC} : {details}")
    elif status == "warn":
        print(f"  {YELLOW}⚠{ENDC} {BOLD}{name}{ENDC} : {details}")
    else:
        print(f"  {RED}✗{ENDC} {BOLD}{name}{ENDC} : {details}")

def run_system_diagnostics():
    print_header("1. SYSTEM & HARDWARE INFO")
    
    # OS
    os_name = platform.system()
    os_release = platform.release()
    print(f"  • OS: {os_name} {os_release} ({platform.machine()})")
    
    # CPU
    cpu_info = platform.processor() or "Unknown CPU"
    print(f"  • Processor: {cpu_info}")
    
    # RAM (approximate for linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            mem = f.readline().split()
            if len(mem) >= 2:
                total_gb = round(int(mem[1]) / (1024 * 1024), 2)
                print(f"  • Memory: {total_gb} GB RAM")
    except Exception:
        print("  • Memory: Unable to read /proc/meminfo")
        
    # GPU / Torch Check
    try:
        import torch
        print_check("PyTorch Version", "ok", torch.__version__)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_check("CUDA GPU", "ok", f"Available - {device_name} ({mem_gb:.1f} GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_check("Apple MPS GPU", "ok", "Available")
        else:
            print_check("Hardware Acceleration", "warn", "No GPU detected (running on CPU)")
    except ImportError:
        print_check("PyTorch", "fail", "Not installed")

def run_dependency_checks():
    print_header("2. PYTHON ENVIRONMENT & DEPENDENCIES")
    
    # Python Version
    py_ver = platform.python_version()
    ver_split = [int(x) for x in py_ver.split('.')]
    if ver_split[0] >= 3 and ver_split[1] >= 10:
        print_check("Python Version", "ok", f"{py_ver} (>= 3.10 standard satisfied)")
    else:
        print_check("Python Version", "fail", f"{py_ver} (Python 3.10+ required)")

    # Read requirements
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print_check("requirements.txt", "fail", "Missing in workspace root")
        return

    print("  Checking imports...")
    dependencies = [
        ("ultralytics", "YOLOv8 Engine"),
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("cv2", "OpenCV Image Processing"),
        ("websockets", "WebSocket protocols"),
        ("multipart", "python-multipart form parsing"),
        ("numpy", "Numerical calculations"),
        ("PIL", "Pillow image library"),
        ("aiofiles", "Asynchronous file I/O"),
        ("albumentations", "Data Augmentation suite")
    ]

    all_ok = True
    for package, desc in dependencies:
        try:
            # Adjust package import names
            import_name = package
            if package == "cv2":
                import_name = "cv2"
            elif package == "PIL":
                import_name = "PIL"
            elif package == "multipart":
                import_name = "multipart"
            
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            print_check(f"dependency: {package}", "ok", f"{desc} (v{ver})")
        except ImportError:
            print_check(f"dependency: {package}", "fail", f"{desc} is NOT installed")
            all_ok = False
            
    if all_ok:
        print(f"\n  {GREEN}✓ All core Python packages are correctly installed!{ENDC}")
    else:
        print(f"\n  {RED}✗ Some dependencies are missing. Run: pip install -r requirements.txt{ENDC}")

def run_file_structure_checks():
    print_header("3. WORKSPACE STRUCTURE & STATIC ASSETS")
    
    required_dirs = [
        "static",
        "static/css",
        "static/js",
        "data",
        "models"
    ]
    
    for d in required_dirs:
        p = Path(d)
        if p.exists() and p.is_dir():
            print_check(f"directory: {d}/", "ok", "Exists")
        else:
            print_check(f"directory: {d}/", "warn", "Missing (will be created on startup)")
            
    required_files = [
        "server.py",
        "detector.py",
        "shelf_analyzer.py",
        "capture.py",
        "db.py",
        "static/index.html",
        "static/css/style.css"
    ]
    
    missing_files = []
    for f in required_files:
        p = Path(f)
        if p.exists() and p.is_file():
            print_check(f"file: {f}", "ok", f"{p.stat().st_size / 1024:.1f} KB")
        else:
            print_check(f"file: {f}", "fail", "MISSING")
            missing_files.append(f)
            
    if not missing_files:
        print(f"\n  {GREEN}✓ Core project files verified successfully.{ENDC}")
    else:
        print(f"\n  {RED}✗ Critical files are missing! Dashboard might not function.{ENDC}")

def run_database_diagnostics():
    print_header("4. SQLITE DATABASE CONFIGURATION")
    
    db_path = Path("data/compliance.db")
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
    try:
        from db import init_db, DB_PATH
        print_check("db.py import", "ok", "Success")
    except ImportError:
        print_check("db.py import", "fail", "Cannot import db module")
        return
        
    try:
        # Initialize if not there
        init_db()
        print_check("Database init", "ok", f"Initialized {DB_PATH}")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ["compliance_events", "hourly_stats"]
        for t in expected_tables:
            if t in tables:
                # Count records
                cursor.execute(f"SELECT COUNT(*) FROM {t}")
                count = cursor.fetchone()[0]
                print_check(f"table: {t}", "ok", f"Healthy ({count} records)")
            else:
                print_check(f"table: {t}", "fail", "MISSING TABLE")
                
        conn.close()
    except Exception as e:
        print_check("Database connection", "fail", str(e))

def run_detector_benchmark():
    print_header("5. DETECTOR INFERENCE BENCHMARK")
    
    # Check if we can import ShelfDetector
    try:
        from detector import ShelfDetector
        print_check("detector.py import", "ok", "Success")
    except ImportError:
        print_check("detector.py import", "fail", "Cannot import ShelfDetector from detector.py")
        return
        
    # Check for weights
    weights = Path("yolov8n.pt")
    if weights.exists():
        print_check("YOLOv8 Weights", "ok", f"Found {weights.name} ({weights.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print_check("YOLOv8 Weights", "warn", "yolov8n.pt not found in root. Will auto-download or run in simulator fallback.")

    # Initialize detector
    try:
        print("  Initializing detector (this may take a few seconds)...")
        start_init = time.time()
        detector = ShelfDetector(model_path="yolov8n.pt")
        init_time = time.time() - start_init
        
        mode_desc = "SIMULATION Mode" if detector.simulation_mode else f"YOLOv8 active ({detector.device})"
        print_check("Detector setup", "ok", f"Initialized in {init_time:.2f}s [{mode_desc}]")
        
        # Benchmark with a dummy image
        print("  Benchmarking inference speed...")
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            detector.detect(dummy_frame)
            
        # Time trials
        latencies = []
        for _ in range(10):
            t0 = time.time()
            detector.detect(dummy_frame)
            latencies.append((time.time() - t0) * 1000)
            
        avg_latency = sum(latencies) / len(latencies)
        fps = 1000 / avg_latency
        
        status = "ok" if avg_latency < 100 else "warn"
        print_check("Inference Latency", status, f"Avg {avg_latency:.2f} ms ({fps:.1f} FPS on dummy 1280x720 frame)")
        
    except Exception as e:
        print_check("Detector benchmark", "fail", f"Benchmark failed: {e}")

def main():
    print(f"\n{BOLD}{GREEN}============================================================{ENDC}")
    print(f"{BOLD}{GREEN}     AETHER-SHELF AI RETAIL SHELF MONITOR DIAGNOSTICS      {ENDC}")
    print(f"{BOLD}{GREEN}============================================================{ENDC}")
    
    run_system_diagnostics()
    run_dependency_checks()
    run_file_structure_checks()
    run_database_diagnostics()
    run_detector_benchmark()
    
    print_header("DIAGNOSTICS COMPLETE")
    print(f"  🚀 {BOLD}To start the server:{ENDC}")
    print(f"     uvicorn server:app --host 0.0.0.0 --port 8000 --reload")
    print(f"  🌐 {BOLD}To expose externally:{ENDC}")
    print(f"     ./tunnel.sh")
    print(f"{BOLD}{CYAN}{'='*60}{ENDC}\n")

if __name__ == "__main__":
    main()
