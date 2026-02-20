"""
generate_dataset.py
-------------------
Generates a synthetic lung CT scan dataset with 700 images across 5 classes.
Each class uses distinct visual patterns to simulate realistic differences
between Normal and cancerous lung tissue types.

Classes:
  0 - Normal Lung
  1 - Adenocarcinoma
  2 - Squamous Cell Carcinoma
  3 - Large Cell Carcinoma
  4 - Small Cell Lung Cancer
"""

import os
import numpy as np
import cv2
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE   = 224          # pixels (width × height)
TOTAL_IMGS = 700          # total images to generate
N_CLASSES  = 5
PER_CLASS  = TOTAL_IMGS // N_CLASSES   # 140 per class
DATA_DIR   = "dataset"

CLASS_NAMES = [
    "Normal",
    "Adenocarcinoma",
    "Squamous_Cell_Carcinoma",
    "Large_Cell_Carcinoma",
    "Small_Cell_Lung_Cancer",
]

np.random.seed(42)


# ── Helper: draw lung outline ─────────────────────────────────────────────────
def draw_lung_outline(img, alpha=0.35):
    """Overlay a faint elliptical lung boundary on a grayscale image."""
    h, w = img.shape[:2]
    overlay = img.copy()
    # Left lobe
    cv2.ellipse(overlay, (w // 3, h // 2),    (w // 5, h // 3),  0, 0, 360, 180, 1)
    # Right lobe
    cv2.ellipse(overlay, (2 * w // 3, h // 2), (w // 5, h // 3), 0, 0, 360, 180, 1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# ── Per-class image generators ────────────────────────────────────────────────
def make_normal(idx):
    """Uniform grey background, faint texture – no nodule."""
    base = np.random.randint(40, 80, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (7, 7), 2)
    # Soft vascular streaks
    for _ in range(np.random.randint(3, 8)):
        pt1 = (np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE))
        pt2 = (np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE))
        cv2.line(base, pt1, pt2, int(np.random.randint(60, 110)), 1)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return draw_lung_outline(img)


def make_adenocarcinoma(idx):
    """Peripheral ground-glass opacity with a soft central nodule."""
    base = np.random.randint(30, 65, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    # Ground-glass haze
    haze = np.zeros_like(base)
    cx, cy = np.random.randint(50, 174, 2)
    cv2.circle(haze, (int(cx), int(cy)), np.random.randint(30, 55), 90, -1)
    haze = cv2.GaussianBlur(haze, (41, 41), 12)
    base = np.clip(base.astype(np.int32) + haze, 0, 255).astype(np.uint8)
    # Solid core
    cv2.circle(base, (int(cx), int(cy)), np.random.randint(8, 16), 200, -1)
    base = cv2.GaussianBlur(base, (5, 5), 1)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return draw_lung_outline(img)


def make_squamous(idx):
    """Central mass near hilum with irregular spiculated border."""
    base = np.random.randint(35, 70, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    # Hilar nodule
    cx, cy = int(IMG_SIZE * 0.5 + np.random.randint(-20, 20)), \
             int(IMG_SIZE * 0.5 + np.random.randint(-20, 20))
    r = np.random.randint(18, 32)
    cv2.circle(base, (cx, cy), r, 210, -1)
    # Spicules
    for _ in range(np.random.randint(6, 14)):
        angle  = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(r + 5, r + 25)
        ex = int(cx + length * np.cos(angle))
        ey = int(cy + length * np.sin(angle))
        cv2.line(base, (cx, cy), (ex, ey), 190, 1)
    base = cv2.GaussianBlur(base, (3, 3), 0.8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return draw_lung_outline(img)


def make_large_cell(idx):
    """Large peripheral irregular mass with necrotic core (dark centre)."""
    base = np.random.randint(30, 60, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cx, cy = np.random.randint(60, 164, 2)
    r_outer = np.random.randint(28, 48)
    r_inner = r_outer // 2
    # Outer mass
    cv2.ellipse(base, (int(cx), int(cy)),
                (r_outer, int(r_outer * np.random.uniform(0.7, 1.0))),
                np.random.randint(0, 45), 0, 360, 220, -1)
    # Necrotic core
    cv2.circle(base, (int(cx), int(cy)), r_inner, 25, -1)
    base = cv2.GaussianBlur(base, (5, 5), 1.2)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return draw_lung_outline(img)


def make_small_cell(idx):
    """Multiple small clustered nodules with mediastinal involvement."""
    base = np.random.randint(35, 65, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    n_nodules = np.random.randint(4, 9)
    # Cluster around mediastinum
    for _ in range(n_nodules):
        cx = int(IMG_SIZE // 2 + np.random.randint(-40, 40))
        cy = int(IMG_SIZE // 2 + np.random.randint(-40, 40))
        r  = np.random.randint(5, 14)
        cv2.circle(base, (cx, cy), r, np.random.randint(170, 230), -1)
    base = cv2.GaussianBlur(base, (3, 3), 0.6)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return draw_lung_outline(img)


GENERATORS = [
    make_normal,
    make_adenocarcinoma,
    make_squamous,
    make_large_cell,
    make_small_cell,
]


# ── Main generation loop ──────────────────────────────────────────────────────
def generate():
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(DATA_DIR) / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        gen_fn = GENERATORS[cls_idx]
        for i in range(PER_CLASS):
            img = gen_fn(i)
            # Apply slight random augmentation for variety
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            if np.random.rand() > 0.5:
                M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2),
                                            np.random.uniform(-15, 15), 1.0)
                img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
            # Add mild Gaussian noise
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            filename = cls_dir / f"{cls_name}_{i:04d}.png"
            cv2.imwrite(str(filename), img)

        print(f"  [{cls_idx + 1}/{N_CLASSES}] {cls_name}: {PER_CLASS} images saved → {cls_dir}")

    total = PER_CLASS * N_CLASSES
    print(f"\nDataset ready: {total} images in '{DATA_DIR}/'")


if __name__ == "__main__":
    print("Generating synthetic lung CT dataset …")
    generate()
