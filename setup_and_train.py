#!/usr/bin/env python3
"""
setup_and_train.py
------------------
ONE-CLICK setup script.

Runs in order:
  1. Install Python dependencies
  2. Generate synthetic dataset (700 images, 5 classes)
  3. Train the CNN model and save lung_cancer_model.h5
  4. Launch the Streamlit web interface

Usage:
    python setup_and_train.py
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd, description=""):
    print(f"\n{'─'*55}")
    print(f"  {description}")
    print(f"{'─'*55}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed (exit code {result.returncode}): {description}")
        sys.exit(result.returncode)


def main():
    print("\n" + "="*55)
    print("  Lung Cancer Detection — One-Click Setup & Train")
    print("="*55)

    # 1. Install dependencies
    run(
        f"{sys.executable} -m pip install -r requirements.txt -q",
        "Step 1/4 — Installing dependencies",
    )

    # 2. Generate dataset (skip if already exists)
    dataset_exists = os.path.isdir(os.path.join(ROOT, "dataset", "Normal"))
    if dataset_exists:
        print("\n  [SKIP] Dataset already exists — skipping generation.")
    else:
        run(
            f"{sys.executable} generate_dataset.py",
            "Step 2/4 — Generating synthetic dataset (700 images)",
        )

    # 3. Train model (skip if already exists)
    model_exists = os.path.exists(os.path.join(ROOT, "lung_cancer_model.h5"))
    if model_exists:
        print("\n  [SKIP] Model already trained — skipping training.")
        print("         Delete lung_cancer_model.h5 to retrain.")
    else:
        run(
            f"{sys.executable} train.py",
            "Step 3/4 — Training CNN model",
        )

    # 4. Launch Streamlit
    print("\n" + "="*55)
    print("  Step 4/4 — Launching Streamlit app")
    print("="*55)
    print("\n  App will be available at: http://localhost:8501\n")
    os.execv(sys.executable,
             [sys.executable, "-m", "streamlit", "run",
              os.path.join(ROOT, "app.py"),
              "--server.port=8501",
              "--server.address=0.0.0.0",
              "--server.headless=true"])


if __name__ == "__main__":
    main()
