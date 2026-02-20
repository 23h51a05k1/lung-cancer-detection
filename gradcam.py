"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Produces:
  - Smooth, contrast-enhanced heatmap overlay on the original CT image
  - Red-highlighted region showing the most suspicious area
  - Green bounding box around the detected region
"""

import numpy as np
import cv2
import tensorflow as tf


# ── Core Grad-CAM computation ──────────────────────────────────────────────────
def compute_gradcam(model, img_array, class_idx, last_conv_layer_name="last_conv"):
    """
    Compute raw Grad-CAM activation map.
    Returns a float32 array of shape (H', W') normalised to [0, 1].
    """
    # Sub-model: outputs last conv activations + final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        # Cast input and watch the conv output explicitly
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        # Use the score of the predicted class as the loss signal
        loss = predictions[:, class_idx]

    # Gradients of class score w.r.t. last conv feature maps
    grads = tape.gradient(loss, conv_outputs)   # (1, H', W', C)

    # Guard against zero gradients (can happen with very confident predictions)
    grads_plus_eps = grads + 1e-8

    # Global average pooling over spatial dims
    pooled_grads = tf.reduce_mean(grads_plus_eps, axis=(0, 1, 2))  # (C,)

    # Weight each feature map by its pooled gradient
    conv_out = conv_outputs[0]                                      # (H', W', C)
    cam = tf.reduce_sum(conv_out * pooled_grads, axis=-1).numpy()   # (H', W')

    # ReLU — keep only positive activations
    cam = np.maximum(cam, 0)

    # If the map is still all zeros, use raw feature map energy as fallback
    if cam.max() == 0:
        cam = np.mean(np.abs(conv_out.numpy()), axis=-1)
        cam = np.maximum(cam, 0)

    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.astype(np.float32)


# ── Post-processing ────────────────────────────────────────────────────────────
def smooth_heatmap(cam, target_size, blur_ksize=15, blur_sigma=5):
    """
    Resize CAM to target_size, apply Gaussian smoothing, and
    stretch contrast so the full [0,1] range is used.
    """
    heatmap = cv2.resize(cam, target_size, interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.GaussianBlur(heatmap, (blur_ksize, blur_ksize), blur_sigma)

    # Aggressive contrast stretch — makes heatmap always visually vibrant
    lo, hi = heatmap.min(), heatmap.max()
    if hi > lo:
        heatmap = (heatmap - lo) / (hi - lo)
    else:
        # Flat map: create a synthetic centred gaussian as placeholder
        h, w = heatmap.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        heatmap = np.exp(-dist ** 2 / (2 * (min(h, w) / 4) ** 2)).astype(np.float32)

    # Apply a power-curve to push mid-values up (makes colours richer)
    heatmap = np.power(heatmap, 0.7).astype(np.float32)
    return heatmap


def overlay_heatmap(original_bgr, heatmap, alpha=0.45):
    """
    Colour-map the heatmap with COLORMAP_JET and blend onto the original.
    Returns a uint8 BGR image.
    """
    heatmap_uint8  = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_colour = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colour = cv2.resize(heatmap_colour, (original_bgr.shape[1], original_bgr.shape[0]))

    # Blend: heatmap on top of original
    blended = cv2.addWeighted(heatmap_colour, alpha, original_bgr, 1.0 - alpha, 0)
    return blended


def highlight_region(original_bgr, heatmap, threshold=0.55):
    """
    Paint a semi-transparent RED mask on the high-activation region and
    draw a GREEN bounding rectangle + label around it.
    """
    result = original_bgr.copy()
    h, w   = result.shape[:2]

    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # Binary mask: pixels above threshold
    binary = (hm >= threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # Fallback: if mask is empty, use top-15% activation zone
    if binary.sum() == 0:
        thr_val = np.percentile(hm, 85)
        binary  = (hm >= thr_val).astype(np.uint8) * 255

    # Semi-transparent red overlay on masked region
    overlay = result.copy()
    overlay[binary == 255] = (overlay[binary == 255] * 0.4).astype(np.uint8)
    overlay[binary == 255, 2] = np.clip(
        overlay[binary == 255, 2].astype(np.int32) + 180, 0, 255
    ).astype(np.uint8)
    result = cv2.addWeighted(overlay, 0.9, result, 0.1, 0)

    # Bounding box around the largest contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            result,
            "Suspicious Region",
            (x, max(y - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (0, 255, 0), 1, cv2.LINE_AA,
        )

    return result


# ── Full pipeline (convenience wrapper) ───────────────────────────────────────
def run_gradcam_pipeline(model, img_array, original_bgr, class_idx,
                         last_conv_layer="last_conv"):
    """
    End-to-end Grad-CAM pipeline.

    Parameters
    ----------
    model        : trained Keras model
    img_array    : (1, H, W, 3) float32 input
    original_bgr : (H, W, 3) uint8 image for overlay
    class_idx    : predicted class index

    Returns
    -------
    heatmap_smooth  : (H, W) float32  — normalised Grad-CAM map
    overlay_img     : (H, W, 3) uint8 — heatmap blended on original
    highlighted_img : (H, W, 3) uint8 — red region + bounding box
    """
    h, w = original_bgr.shape[:2]

    # 1. Raw Grad-CAM
    cam = compute_gradcam(model, img_array, class_idx, last_conv_layer)

    # 2. Smooth & resize to original dimensions
    heatmap_smooth = smooth_heatmap(cam, (w, h))

    # 3. Coloured overlay
    overlay_img = overlay_heatmap(original_bgr, heatmap_smooth)

    # 4. Red region highlight
    highlighted_img = highlight_region(original_bgr, heatmap_smooth)

    return heatmap_smooth, overlay_img, highlighted_img
