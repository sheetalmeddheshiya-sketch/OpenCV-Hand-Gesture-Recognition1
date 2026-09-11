"""Interactive Streamlit dashboard for the trained hand gesture model."""

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

from gesture_utils import extract_features, hand_present

ROOT = Path(__file__).resolve().parent
MODEL_PATH = next(
    (
        candidate
        for candidate in (
            ROOT / "models" / "gesture_model.joblib",
            ROOT / "gesture_model.joblib",
        )
        if candidate.exists()
    ),
    ROOT / "models" / "gesture_model.joblib",
)
METADATA_PATH = next(
    (
        candidate
        for candidate in (
            ROOT / "models" / "gesture_model.json",
            ROOT / "gesture_model.json",
        )
        if candidate.exists()
    ),
    ROOT / "models" / "gesture_model.json",
)

st.set_page_config(
    page_title="GestureAI Dashboard",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1220; color: #e5edf8; }
    [data-testid="stSidebar"] { background: #111b2e; border-right: 1px solid #233451; }
    .hero { padding: 28px 32px; border-radius: 22px; background: linear-gradient(120deg,#172b4d,#153b53); margin-bottom: 22px; }
    .hero h1 { color: #f4f8ff; font-size: 2.5rem; margin: 0; }
    .hero p { color: #a9c0dd; margin: 8px 0 0; font-size: 1.05rem; }
    .metric { background: #121f34; border: 1px solid #253b5b; border-radius: 16px; padding: 18px; }
    .metric-label { color: #8ea6c5; font-size: .85rem; }
    .metric-value { color: #f4f8ff; font-size: 1.7rem; font-weight: 700; margin-top: 5px; }
    .section-title { color: #eaf2ff; font-size: 1.25rem; font-weight: 650; margin: 18px 0 12px; }
    .gesture { background: #121f34; border-radius: 12px; padding: 12px; border: 1px solid #233a5a; text-align: center; }
    .gesture b { color: #eef5ff; }
    .gesture small { color: #8ea6c5; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file is missing from the deployed app. "
            "Upload gesture_model.joblib and gesture_model.json "
            "to the repository root (or put them inside models/) "
            "to the GitHub repository, then redeploy. "
            "Training is not run automatically on Streamlit Cloud."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def predict_image(model, image: Image.Image):
    frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if not hand_present(frame):
        return None, 0.0, []
    features = extract_features(frame).reshape(1, -1)
    label = model.predict(features)[0]
    scores = model.decision_function(features)[0]
    probabilities = np.exp(scores - np.max(scores))
    probabilities = probabilities / probabilities.sum()
    order = np.argsort(probabilities)[::-1]
    return label, float(probabilities[order[0]]), [
        (model.classes_[index], float(probabilities[index])) for index in order
    ]


try:
    model = load_model()
    metadata = load_metadata()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

classes = list(model.classes_)
st.markdown(
    '<div class="hero"><h1>✋ GestureAI</h1>'
    '<p>OpenCV + Machine Learning hand gesture recognition dashboard</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Control panel")
    st.caption("Upload a hand image to run the trained classifier.")
    st.divider()
    st.markdown("### Model status")
    st.success("● Model ready")
    st.caption(f"File: {MODEL_PATH.name}")
    st.caption("Feature extractor: OpenCV HOG")
    st.caption("Classifier: Linear SVM")

metrics = st.columns(4)
metric_values = [
    ("MODEL ACCURACY", f"{metadata.get('validation_accuracy', 0):.0%}"),
    ("GESTURE CLASSES", str(len(classes))),
    ("TRAINING SAMPLES", f"{metadata.get('samples', 0):,}"),
    ("INFERENCE MODE", "REAL-TIME"),
]
for column, (label, value) in zip(metrics, metric_values):
    column.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown('<div class="section-title">Try a prediction</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Input source",
        ["Webcam", "Upload image"],
        horizontal=True,
        label_visibility="collapsed",
    )
    camera_image = None
    upload = None
    if input_mode == "Webcam":
        st.caption("Camera के सामने हाथ रखें और image capture करें।")
        camera_image = st.camera_input("Capture hand gesture")
    else:
        upload = st.file_uploader("Upload a hand image", type=["png", "jpg", "jpeg"])

    image_source = camera_image if camera_image is not None else upload
    if image_source:
        image = Image.open(image_source)
        label, confidence, ranking = predict_image(model, image)
        image_col, result_col = st.columns([1, 1])
        with image_col:
            st.image(
                image,
                caption="Camera capture" if camera_image else "Input image",
                use_container_width=True,
            )
        with result_col:
            if label is None:
                st.warning("No hand detected. Please place your hand in front of the camera.")
            else:
                st.markdown("#### Detected gesture")
                st.markdown(f"## `{label}`")
                st.progress(min(confidence, 1.0), text=f"Confidence {confidence:.1%}")
                st.caption("Confidence is normalized from the SVM decision scores.")
                st.markdown("#### Ranking")
                for gesture, score in ranking[:5]:
                    st.write(f"**{gesture}** — {score:.1%}")
                    st.progress(score)
    else:
        message = (
            "Capture a webcam image to see the prediction here."
            if input_mode == "Webcam"
            else "Upload a PNG/JPG hand image to see the prediction here."
        )
        st.info(message)

with right:
    st.markdown('<div class="section-title">Supported gestures</div>', unsafe_allow_html=True)
    rows = [classes[index:index + 2] for index in range(0, len(classes), 2)]
    for row in rows:
        gesture_columns = st.columns(2)
        for column, gesture in zip(gesture_columns, row):
            column.markdown(
                f'<div class="gesture"><b>✋ {gesture}</b><br>'
                '<small>Available class</small></div>',
                unsafe_allow_html=True,
            )
        st.write("")
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        "1. Image is converted to a hand silhouette with OpenCV.\n"
        "2. HOG features capture the hand shape.\n"
        "3. Linear SVM predicts the gesture class."
    )

st.divider()
st.caption("Webcam capture और image upload दोनों इसी dashboard में उपलब्ध हैं।")
