"""
=============================================================================
             🌿 PLANT DISEASE DETECTION - STREAMLIT GUI
=============================================================================
Modern, production-grade web interface for plant disease classification
using EfficientNetV2-B0 + Transfer Learning in TensorFlow/Keras.
=============================================================================
"""

import sys
from pathlib import Path
from PIL import Image
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Safe import of prediction engine
try:
    from src.predict import load_trained_model, predict_image, DEFAULT_MODEL_PATH
except ImportError as err:
    st.error(f"Failed to load project prediction module: {err}")
    st.stop()


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI with high contrast in dark/light themes
st.markdown("""
    <style>
    /* Header typography with high contrast for dark & light mode */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #52b788;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #95d5b2;
        margin-bottom: 1.8rem;
        font-weight: 500;
    }
    .badge-tag {
        background-color: #1b4332;
        color: #74c69d !important;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
        border: 1px solid #2d6a4f;
    }
    /* Explicit contrast rules for custom cards in dark & light themes */
    .prediction-card {
        background-color: #ffffff;
        color: #212529 !important;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e9ecef;
        border-left: 8px solid #2d6a4f;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .prediction-card * {
        color: #212529 !important;
    }
    .prediction-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1b4332 !important;
        margin-bottom: 0.4rem;
    }
    .confidence-badge {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2d6a4f !important;
    }
    .unrecognized-card {
        background-color: #fff5f5;
        color: #742a2a !important;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #feb2b2;
        border-left: 8px solid #e53e3e;
        margin-bottom: 20px;
    }
    .unrecognized-card * {
        color: #742a2a !important;
    }
    .unrecognized-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #c53030 !important;
        margin-bottom: 0.5rem;
    }
    .fun-disclaimer-box {
        background-color: #f8f9fa;
        color: #212529 !important;
        border-radius: 10px;
        padding: 18px 22px;
        border: 1px solid #dee2e6;
        border-left: 6px solid #2e7d32;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-top: 30px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .fun-disclaimer-box * {
        color: #212529 !important;
    }
    .fun-disclaimer-box small, .fun-disclaimer-box i {
        color: #495057 !important;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR CONTENT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=64)
    st.title("🌿 Plant Disease AI")
    st.markdown('<div class="badge-tag">Deep Learning AI System</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("ℹ️ Model Architecture")
    st.markdown("""
    - **Architecture**: EfficientNetV2-B0
    - **Methodology**: Transfer Learning
    - **Dataset**: PlantVillage (6,498 images)
    - **Input Resolution**: 224 × 224 pixels (RGB)
    - **Classes (5)**:
      1. Tomato Healthy
      2. Tomato Early Blight
      3. Tomato Late Blight
      4. Potato Early Blight
      5. Potato Late Blight
    """)

    st.markdown("---")
    st.caption("Developed using TensorFlow/Keras and Streamlit.")


# -----------------------------------------------------------------------------
# MAIN CONTENT HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated leaf image classification using EfficientNetV2-B0 + Transfer Learning</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODEL LOADING WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_model():
    """Loads and caches model to prevent reload delay on UI interaction."""
    try:
        return load_trained_model()
    except Exception as e:
        return None, str(e)

model_tuple = get_cached_model()
model, model_err = model_tuple if isinstance(model_tuple, tuple) and model_tuple[0] is not None else (None, "Model artifact missing")

if model is None:
    st.warning("⚠️ **Trained Model Artifact Not Found**")
    st.info("""
    The trained EfficientNetV2-B0 Keras model artifact (`models/plant_disease_model.keras`) is currently missing.

    **To train the model:**
    ```bash
    python src/train.py
    ```
    """)
    st.stop()


# -----------------------------------------------------------------------------
# IMAGE UPLOAD & INFERENCE SECTION
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a leaf photo of a tomato or potato plant to analyze disease symptoms."
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("🖼️ Uploaded Image Preview")
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption=f"File: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading image file: {e}")
            st.stop()

    with col2:
        st.subheader("🔍 Classification Results")

        with st.spinner("Analyzing image features using EfficientNetV2-B0..."):
            try:
                # Perform prediction with safe non-leaf rejection
                result = predict_image(model, model_tuple[1], image)
                
                is_unrecognized = result.get("is_unrecognized", False)
                unrecognized_reason = result.get("unrecognized_reason", "")
                pred_class = result["class_name"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]

                if is_unrecognized:
                    # Safe rejection UI for non-leaf or unrelated images
                    st.markdown(f"""
                        <div class="unrecognized-card">
                            <div class="unrecognized-title">⚠️ Image Not Recognized</div>
                            <p style="color: #742a2a; margin: 0; font-size: 1.05rem;">
                                {unrecognized_reason}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.info("Please upload a clear leaf image of a supported plant (Tomato or Potato).")
                else:
                    # Professional result display for valid leaf images
                    st.markdown(f"""
                        <div class="prediction-card">
                            <div style="font-size: 0.9rem; color: #6c757d; font-weight: 600; text-transform: uppercase;">Prediction</div>
                            <div class="prediction-title">{pred_class}</div>
                            <div style="margin-top: 10px;">
                                <span style="font-size: 0.9rem; color: #6c757d; font-weight: 600; text-transform: uppercase;">Confidence: </span>
                                <span class="confidence-badge">{confidence * 100:.1f}%</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    if "healthy" in pred_class.lower():
                        st.success(f"🌱 **Plant Status**: The model classifies this leaf as **{pred_class}**.")
                    else:
                        st.error(f"⚠️ **Plant Status**: Disease identified as **{pred_class}**.")

                    # Class Probabilities Breakdown
                    st.markdown("#### 📊 Probability Distribution")
                    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    
                    for cls_name, prob in sorted_probs:
                        pct = prob * 100
                        st.write(f"**{cls_name}** ({pct:.1f}%)")
                        st.progress(min(max(float(prob), 0.0), 1.0))

            except Exception as eval_err:
                st.error(f"An error occurred during classification: {eval_err}")

    # Disclaimer Block
    st.markdown("""
        <div class="fun-disclaimer-box">
            🌱 <b>AI says:</b> “Looks like Tomato Late Blight!”<br>
            👨‍🌾 <b>Farmer:</b> “Bro, are you sure?”<br>
            🤖 <b>AI:</b> “I’m a CNN, not an agricultural scientist. 😭”<br><br>
            <small><i>For educational purposes only. Please consult an agricultural expert before making treatment decisions.</i></small>
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Upload a leaf image to receive automated disease classification.")
