import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os

st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="centered")

st.title("♻️ RecycleVision — Garbage Classifier")
st.write(
    "Upload an image of waste, and the model will classify it into the appropriate category."
)

# Auto-detect model file (.keras or .h5)
MODEL_DIR = "artifacts"
LABEL_MAP = os.path.join(MODEL_DIR, "label_map.json")


def get_model_path():
    keras_path = os.path.join(MODEL_DIR, "best_model.keras")
    h5_path = os.path.join(MODEL_DIR, "best_model.h5")
    if os.path.exists(keras_path):
        return keras_path
    elif os.path.exists(h5_path):
        return h5_path
    return keras_path  # fallback default


DEFAULT_MODEL = get_model_path()


@st.cache_resource
def load_model(path):
    # Allow loading both .keras and .h5 formats
    custom_objects = {
        "TrueDivide": tf.keras.layers.Lambda(lambda x: tf.math.truediv(x, 1.0))
    }
    try:
        return tf.keras.models.load_model(path, custom_objects=custom_objects)
    except Exception as e:
        st.warning(f"Standard load failed ({e}). Trying without custom objects...")
        return tf.keras.models.load_model(path)


@st.cache_data
def load_label_map(path):
    with open(path) as f:
        lm = json.load(f)
    if isinstance(lm, dict):
        classes = [lm[str(i)] for i in range(len(lm))]
    else:
        classes = lm
    return classes


def preprocess(img: Image.Image, img_size=224):
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
with col2:
    model_path = st.text_input("Model path", value=DEFAULT_MODEL)

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Input", use_column_width=True)

    if not os.path.exists(model_path) or not os.path.exists(LABEL_MAP):
        st.error("Model or label_map not found. Train the model first, then try again.")
    else:
        with st.spinner("Predicting..."):
            model = load_model(model_path)
            classes = load_label_map(LABEL_MAP)
            x = preprocess(image, img_size=model.input_shape[1])
            probs = model.predict(x, verbose=0)[0]
            top3_idx = probs.argsort()[-3:][::-1]

            st.subheader(f"Prediction: **{classes[int(top3_idx[0])]}**")
            st.write("Confidence:", float(probs[top3_idx[0]]))
            st.write("Top-3 Predictions:")
            for idx in top3_idx:
                st.write(f"- {classes[int(idx)]}: {float(probs[idx]):.4f}")
