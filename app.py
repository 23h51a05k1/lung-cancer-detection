"""
app.py
------
Streamlit web interface for the Lung Cancer Detection system.

Features:
  - Upload a CT scan (PNG / JPG)
  - Full preprocessing pipeline
  - CNN inference → class + confidence score
  - Grad-CAM heatmap overlay
  - Red-region highlighting with bounding box
  - Cancer type description panel
  - Download results
"""

import os
import io
import sys
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CLASS_NAMES, CLASS_DESCRIPTIONS, IMG_SIZE
from gradcam import run_gradcam_pipeline, overlay_heatmap, highlight_region
from train import preprocess_image

# ── Constants ──────────────────────────────────────────────────────────────────
_APP_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_APP_DIR, "lung_cancer_model.h5")

# ── Auto-download model from Hugging Face Hub if not present ──────────────────
HF_REPO_ID   = os.environ.get("HF_REPO_ID", "")          # set in Streamlit secrets
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_FILENAME  = "lung_cancer_model.h5"

def ensure_model():
    """Download the model from HF Hub when running in the cloud."""
    if os.path.exists(MODEL_PATH):
        return True
    if not HF_REPO_ID:
        return False
    try:
        from huggingface_hub import hf_hub_download
        st.info("Downloading model from Hugging Face Hub — please wait…")
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME,
                               local_dir=_APP_DIR)
        if path != MODEL_PATH:
            import shutil
            shutil.copy(path, MODEL_PATH)
        return True
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return False

CLASS_COLORS = {
    "Normal":                  "#27ae60",
    "Adenocarcinoma":          "#e67e22",
    "Squamous_Cell_Carcinoma": "#e74c3c",
    "Large_Cell_Carcinoma":    "#8e44ad",
    "Small_Cell_Lung_Cancer":  "#c0392b",
}

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.4rem; font-weight: 700;
    color: #2c3e50; text-align: center; margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1.05rem; color: #7f8c8d;
    text-align: center; margin-bottom: 1.5rem;
}
.result-box {
    background: #f8f9fa; border-radius: 12px;
    padding: 1.2rem 1.6rem; margin-top: 0.8rem;
}
.section-header {
    font-size: 1.1rem; font-weight: 600;
    color: #2c3e50; margin-top: 1rem; margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🫁 Lung Cancer Detection & Visualization</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload a lung CT scan to detect cancer type, '
    'confidence score, and visualise the affected region using Grad-CAM.</p>',
    unsafe_allow_html=True)
st.divider()

# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CNN model …")
def load_model():
    """Load the trained Keras model from disk (cached across reruns)."""
    if not ensure_model():
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.error(
        "**Model not found.**  \n"
        "Either run the setup script locally:\n"
        "```bash\npython setup_and_train.py\n```\n"
        "Or set the `HF_REPO_ID` secret in Streamlit Cloud to auto-download the model."
    )
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**System:** Custom CNN trained on 700 synthetic lung CT images.

**Classes detected:**
- Normal Lung
- Adenocarcinoma
- Squamous Cell Carcinoma
- Large Cell Carcinoma
- Small Cell Lung Cancer

**Visualisation:** Grad-CAM highlights regions that most influenced the prediction.

---
*For research / educational use only. Not a medical diagnostic tool.*
""")
    st.header("⚙️ Settings")
    heatmap_alpha = st.slider("Heatmap opacity",   0.10, 0.90, 0.45, 0.05)
    region_thresh = st.slider("Region threshold",  0.30, 0.90, 0.55, 0.05)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload CT scan image (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded is None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.info("⬆️  Upload a CT scan image above to begin analysis.")
    st.stop()

# ── Image helpers ──────────────────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image → OpenCV BGR uint8."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR uint8 → PIL Image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def img_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode PIL image to bytes for download button."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

# ── Process image ──────────────────────────────────────────────────────────────
try:
    pil_img     = Image.open(uploaded).convert("RGB")
    bgr_orig    = pil_to_bgr(pil_img)
    bgr_display = cv2.resize(bgr_orig, (IMG_SIZE, IMG_SIZE))  # fixed 224×224

    # Preprocess for model
    preprocessed = preprocess_image(bgr_display)              # (224,224,3) float32
    inp_array    = np.expand_dims(preprocessed, 0)            # (1,224,224,3)

    # Predict
    probs     = model.predict(inp_array, verbose=0)[0]        # (5,)
    cls_idx   = int(np.argmax(probs))
    cls_name  = CLASS_NAMES[cls_idx]
    confidence = float(probs[cls_idx]) * 100.0

    # Grad-CAM (base run to get heatmap)
    heatmap_smooth, _, _ = run_gradcam_pipeline(
        model, inp_array, bgr_display, cls_idx
    )

    # Apply sidebar-controlled alpha / threshold
    overlay_img     = overlay_heatmap(bgr_display, heatmap_smooth, alpha=heatmap_alpha)
    highlighted_img = highlight_region(bgr_display, heatmap_smooth, threshold=region_thresh)

except Exception as e:
    st.error(f"Error processing image: {e}")
    st.stop()

# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)

    label_color  = CLASS_COLORS.get(cls_name, "#2c3e50")
    display_name = cls_name.replace("_", " ")

    st.markdown(f"""
<div class="result-box">
    <div style="font-size:1.5rem;font-weight:700;color:{label_color};">{display_name}</div>
    <div style="margin-top:8px;color:#636e72;font-size:0.9rem;">Confidence</div>
    <div style="font-size:1.8rem;font-weight:700;color:#2d3436;">{confidence:.1f}%</div>
    <div style="background:#dfe6e9;border-radius:6px;height:14px;width:100%;margin-top:6px;">
        <div style="width:{confidence:.1f}%;background:{label_color};
                    border-radius:6px;height:14px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Class Probabilities</p>', unsafe_allow_html=True)
    for name, prob in zip(CLASS_NAMES, probs):
        bar_w = float(prob) * 100
        color = CLASS_COLORS.get(name, "#636e72")
        disp  = name.replace("_", " ")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"""
<div style="margin-bottom:6px;">
    <span style="font-size:0.82rem;color:#2d3436;">{disp}</span>
    <div style="background:#dfe6e9;border-radius:6px;height:10px;width:100%;margin-top:3px;">
        <div style="width:{bar_w:.1f}%;background:{color};border-radius:6px;height:10px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)
        with col_b:
            st.markdown(
                f'<div style="font-size:0.82rem;padding-top:16px;">{bar_w:.1f}%</div>',
                unsafe_allow_html=True)

    st.markdown('<p class="section-header">Cancer Description</p>', unsafe_allow_html=True)
    st.info(CLASS_DESCRIPTIONS.get(cls_name, "No description available."))


with col_right:
    st.markdown('<p class="section-header">Visual Analysis</p>', unsafe_allow_html=True)

    img_c1, img_c2, img_c3 = st.columns(3)
    with img_c1:
        st.markdown("**Original CT Scan**")
        st.image(bgr_to_pil(bgr_display), use_column_width=True)
    with img_c2:
        st.markdown("**Grad-CAM Heatmap**")
        st.image(bgr_to_pil(overlay_img), use_column_width=True)
    with img_c3:
        st.markdown("**Affected Region**")
        st.image(bgr_to_pil(highlighted_img), use_column_width=True)

    st.markdown("""
<div style="font-size:0.8rem;color:#636e72;margin-top:6px;">
<b>Heatmap:</b> Red/yellow = high activation, Blue = low activation &nbsp;|&nbsp;
<b>Region:</b> Red overlay = suspected area, Green box = detected boundary
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Download ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Download Results</p>', unsafe_allow_html=True)
dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("⬇ Original",    img_to_bytes(bgr_to_pil(bgr_display)),
                       "ct_original.png",    "image/png")
with dl2:
    st.download_button("⬇ Heatmap",     img_to_bytes(bgr_to_pil(overlay_img)),
                       "ct_heatmap.png",     "image/png")
with dl3:
    st.download_button("⬇ Highlighted", img_to_bytes(bgr_to_pil(highlighted_img)),
                       "ct_highlighted.png", "image/png")

st.markdown(
    "<div style='text-align:center;color:#b2bec3;font-size:0.75rem;margin-top:1rem;'>"
    "Lung Cancer Detection System — For educational/research use only"
    "</div>",
    unsafe_allow_html=True)
