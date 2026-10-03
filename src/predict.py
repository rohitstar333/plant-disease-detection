"""
=============================================================================
             SINGLE IMAGE PREDICTION & INFERENCE ENGINE
=============================================================================
This module provides reusable image preprocessing, model inference, and safe
unknown-image rejection functions using EfficientNetV2-B0 + Transfer Learning
for CLI scripts and the Streamlit web application.
=============================================================================
"""

import json
import io
from pathlib import Path
from typing import Dict, Union, Tuple, List

import numpy as np
from PIL import Image
import tensorflow as tf


# =============================================================================
# PROJECT DEFAULT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "plant_disease_model.keras"
)

DEFAULT_CLASS_NAMES_PATH = (
    PROJECT_ROOT / "models" / "class_names.json"
)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_trained_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    class_names_path: Union[str, Path] = DEFAULT_CLASS_NAMES_PATH
) -> Tuple[tf.keras.Model, List[str]]:
    """
    Load the trained EfficientNetV2-B0 + Transfer Learning model
    and class configuration.

    Returns:
        tuple:
            - trained TensorFlow/Keras model
            - list of class display names
    """

    model_path = Path(model_path)
    class_names_path = Path(class_names_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model file not found at: {model_path}\n"
            f"Please run model training first using: python src/train.py"
        )

    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names configuration file not found at: {class_names_path}"
        )

    # Load trained Keras model
    model = tf.keras.models.load_model(model_path)

    # Load class names
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_config = json.load(f)

    display_names = class_config.get("class_display_names", [])

    if not display_names:
        raise ValueError(
            "Class configuration file contains no display names!"
        )

    return model, display_names


# =============================================================================
# LEAF COLOR & VEGETATION HEURISTIC CHECK
# =============================================================================

def check_leaf_color_ratio(img: Image.Image) -> float:
    """
    Computes proportion of pixels matching plant leaf color characteristics
    (healthy green, chlorotic yellow, or necrotic brown/tan lesion tissue).
    Distinguishes plant tissue from human skin tones, vehicles, and metal objects.
    """
    img_rgb = img.convert("RGB").resize((100, 100))
    arr = np.array(img_rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Condition 1: Healthy or early lesion green tissue (Green dominant)
    is_green = (g > r * 0.85) & (g > b * 1.05) & (g > 25)

    # Condition 2: Chlorotic / Yellow leaf tissue (R and G both high, B low)
    is_yellow = (r > 50) & (g > 50) & (b < r * 0.80) & (g > b * 1.15) & (abs(r - g) < 45)

    # Condition 3: Necrotic / Brown lesion leaf tissue (R > B, G > B, but R and G close, unlike skin)
    is_brown = (r > 35) & (g > 25) & (r > b * 1.15) & (g > b * 0.95) & (r - g < 35) & (r < 220) & (g < 200)

    leaf_mask = is_green | is_yellow | is_brown
    ratio = float(np.mean(leaf_mask))
    return ratio


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(
    image_input: Union[str, Path, Image.Image, bytes],
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, Image.Image]:
    """
    Preprocess an input image for EfficientNetV2-B0.

    EfficientNetV2 preprocessing is handled INSIDE the trained model graph.
    Therefore, this function intentionally passes raw [0, 255] RGB float32 arrays.

    Returns:
        tuple: (batch_array of shape (1, 224, 224, 3), PIL.Image instance)
    """

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Target image path does not exist: {path}")
        pil_img = Image.open(path)
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    pil_img_rgb = pil_img.convert("RGB")
    resized_img = pil_img_rgb.resize(target_size, Image.Resampling.LANCZOS)

    img_array = np.array(resized_img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch, pil_img_rgb


# =============================================================================
# IMAGE PREDICTION WITH SAFE REJECTION
# =============================================================================

def predict_image(
    model: tf.keras.Model,
    class_names: List[str],
    image_input: Union[str, Path, Image.Image, bytes],
    confidence_threshold: float = 0.40
) -> Dict:
    """
    Perform single-image disease classification with non-leaf image rejection.

    Returns:
        Dict containing:
            - class_name: Predicted class name
            - confidence: Float prediction probability (0.0 to 1.0)
            - entropy: Shannon entropy score
            - is_uncertain: Boolean flag for high entropy
            - is_unrecognized: Boolean flag for non-leaf or unrelated images
            - unrecognized_reason: Explanation if rejected
            - leaf_ratio: Measured leaf color proportion
            - probabilities: Dict mapping display class names to probabilities
    """

    img_batch, pil_img = preprocess_image(image_input)

    # 1. Non-leaf heuristic check
    leaf_ratio = check_leaf_color_ratio(pil_img)

    # 2. Model prediction
    raw_predictions = model.predict(img_batch, verbose=0)[0]

    top_idx = int(np.argmax(raw_predictions))
    top_class_name = class_names[top_idx]
    top_confidence = float(raw_predictions[top_idx])

    # 3. Shannon entropy calculation H(p) = -sum(p * log(p))
    eps = 1e-12
    entropy = -float(np.sum(raw_predictions * np.log(raw_predictions + eps)))

    # 4. Uncertainty and Rejection flags
    is_uncertain = (top_confidence < 0.50 or entropy > 1.35)

    # Reject if non-leaf color presence (< 8%) OR extremely low confidence (< 35%) OR high entropy + low top probability
    is_unrecognized = False
    unrecognized_reason = ""

    if leaf_ratio < 0.08:
        is_unrecognized = True
        unrecognized_reason = "This image does not appear to be a supported plant leaf."
    elif top_confidence < 0.35:
        is_unrecognized = True
        unrecognized_reason = "Low classification confidence across all supported disease classes."
    elif entropy > 1.45 and top_confidence < 0.45:
        is_unrecognized = True
        unrecognized_reason = "High prediction uncertainty across multiple disease classes."

    probabilities = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(class_names, raw_predictions)
    }

    return {
        "class_name": top_class_name,
        "confidence": round(top_confidence, 4),
        "entropy": round(entropy, 4),
        "is_uncertain": is_uncertain,
        "is_unrecognized": is_unrecognized,
        "unrecognized_reason": unrecognized_reason,
        "leaf_ratio": round(leaf_ratio, 4),
        "probabilities": probabilities
    }


if __name__ == "__main__":
    print("Prediction module ready for EfficientNetV2-B0 + Transfer Learning.")