import os
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Manufacturing Defect Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "final_manufacturing_defect_resnet50.keras"
)

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

DISPLAY_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale",
    "scratches": "Scratches"
}


# ============================================================
# LIGHT UI
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748b;
        font-size: 18px;
        margin-bottom: 24px;
    }

    .result-card {
        padding: 22px;
        border-radius: 14px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        margin-top: 15px;
    }

    .result-defect {
        font-size: 30px;
        font-weight: 800;
        color: #166534;
        margin-top: 5px;
    }

    .small-note {
        color: #64748b;
        font-size: 14px;
    }

    .process-note {
        color: #475569;
        font-size: 14px;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔍 Manufacturing Defect Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based steel surface defect classification using fine-tuned ResNet50.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error("Unable to load the ResNet50 model.")
    st.info(
        "Place 'final_manufacturing_defect_resnet50.keras' "
        "in the same folder as app.py."
    )
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input":
                tf.keras.applications.resnet50.preprocess_input
        },
        compile=False
    )


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained ResNet50 model.")
    st.exception(e)
    st.stop()


st.success("✅ Fine-tuned ResNet50 model loaded successfully")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload Steel Surface Image")

uploaded_file = st.file_uploader(
    "Drag and drop an image here or browse files",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload a steel surface image to begin defect detection.")
    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

image = Image.open(uploaded_file).convert("RGB")


# ============================================================
# IMAGE + DETECTION BUTTON
# ============================================================

left, right = st.columns([1, 1])

with left:
    st.subheader("🖼️ Uploaded Steel Image")
    st.image(image, use_container_width=True)

    st.caption(
        f"Image size: {image.width} × {image.height} • RGB"
    )


with right:
    st.subheader("🤖 AI Prediction")

    detect_button = st.button(
        "🔍 DETECT DEFECT",
        type="primary",
        use_container_width=True
    )


# ============================================================
# COMPLETE PROCESS INSIDE ONE SMALL TAP / STATUS BOX
# ============================================================

if detect_button:

    # The entire process is intentionally kept inside ONE
    # collapsible status component. No repeated diagrams.
    with st.status(
        "🔄 AI Processing Pipeline",
        expanded=True
    ) as process:

        # ----------------------------------------------------
        # STEP 1 — IMAGE INPUT
        # ----------------------------------------------------

        st.write("📤 **Step 1 — Image received**")
        st.caption(
            "Steel surface image loaded successfully."
        )

        time.sleep(0.25)

        # ----------------------------------------------------
        # STEP 2 — PREPROCESSING
        # ----------------------------------------------------

        st.write("⚙️ **Step 2 — Preprocessing**")
        st.caption(
            "Resizing image to 128 × 128 RGB."
        )

        resized_image = image.resize((128, 128))

        image_array = np.asarray(
            resized_image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        time.sleep(0.25)

        # ----------------------------------------------------
        # STEP 3 — RESNET50 FEATURE EXTRACTION
        # ----------------------------------------------------

        st.write("🧠 **Step 3 — ResNet50 feature extraction**")
        st.caption(
            "Passing the preprocessed image through the "
            "fine-tuned ResNet50 network."
        )

        time.sleep(0.25)

        # ----------------------------------------------------
        # STEP 4 — CLASSIFICATION
        # ----------------------------------------------------

        st.write("🎯 **Step 4 — Classification**")
        st.caption(
            "Calculating probabilities for six defect classes."
        )

        prediction = model.predict(
            image_array,
            verbose=0
        )[0]

        prediction = np.asarray(
            prediction,
            dtype=np.float32
        )

        prediction = np.squeeze(prediction)

        # Normalize output if the model does not already return
        # probabilities.
        if (
            np.any(prediction < 0)
            or
            not np.isclose(
                np.sum(prediction),
                1.0,
                atol=0.01
            )
        ):
            prediction = tf.nn.softmax(
                prediction
            ).numpy()

        time.sleep(0.25)

        # ----------------------------------------------------
        # STEP 5 — RESULT
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(prediction)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            prediction[predicted_index] * 100
        )

        st.write("🏆 **Step 5 — Final result generated**")
        st.caption(
            f"Predicted class: {DISPLAY_NAMES[predicted_class]}"
        )

        process.update(
            label="✅ AI Processing Completed",
            state="complete",
            expanded=False
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    st.markdown(
        f"""
        <div class="result-card">
            <div class="small-note">🏆 AI DETECTION RESULT</div>
            <div class="result-defect">
                {DISPLAY_NAMES[predicted_class]}
            </div>
            <div style="font-size:18px; color:#166534; margin-top:8px;">
                Confidence: <b>{confidence:.2f}%</b>
            </div>
            <div style="color:#15803d; margin-top:10px;">
                ✓ Defect classification completed successfully.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    st.subheader("📊 Class Probabilities")

    for cls, prob in zip(CLASS_NAMES, prediction):

        percentage = float(prob * 100)

        col1, col2 = st.columns([5, 1])

        with col1:
            st.write(
                f"**{DISPLAY_NAMES[cls]}**"
            )

            st.progress(
                float(np.clip(prob, 0, 1))
            )

        with col2:
            st.write(
                f"**{percentage:.2f}%**"
            )

    # ========================================================
    # PROCESSING SUMMARY
    # ========================================================

    st.subheader("🔬 Processing Summary")

    summary_cols = st.columns(5)

    summary_data = [
        ("📤", "INPUT", "Steel Image"),
        ("⚙️", "PREPROCESS", "128 × 128 RGB"),
        ("🧠", "RESNET50", "Feature Extraction"),
        ("🎯", "CLASSIFIER", "6 Classes"),
        ("🏆", "RESULT", DISPLAY_NAMES[predicted_class])
    ]

    for col, (icon, title, value) in zip(
        summary_cols,
        summary_data
    ):
        with col:
            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    padding:16px 8px;
                    border:1px solid #e2e8f0;
                    border-radius:12px;
                    background:#f8fafc;
                    min-height:105px;
                ">
                    <div style="font-size:25px;">{icon}</div>
                    <div style="
                        font-size:12px;
                        font-weight:700;
                        color:#64748b;
                        margin-top:5px;
                    ">{title}</div>
                    <div style="
                        font-size:14px;
                        font-weight:700;
                        color:#172554;
                        margin-top:5px;
                    ">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Manufacturing Defect Detection • "
    "Fine-tuned ResNet50 • Six Steel Surface Defect Classes"
)
