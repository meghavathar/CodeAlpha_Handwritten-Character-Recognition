import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import json


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("✍️ Handwritten Character Recognition")

st.write(
    "Upload an image of a handwritten character and "
    "the CNN model will predict the character."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "handwritten_character_model.keras"
    )

    return model


# ============================================================
# LOAD LABELS
# ============================================================

@st.cache_data
def load_labels():

    with open("labels.json", "r") as file:

        labels = json.load(file)

    return labels


model = load_model()
labels = load_labels()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten character image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        width=250
    )


    # --------------------------------------------------------
    # Convert image to grayscale
    # --------------------------------------------------------

    image = image.convert("L")


    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    image = image.resize(
        (28, 28)
    )


    # --------------------------------------------------------
    # Invert image
    # --------------------------------------------------------

    image = ImageOps.invert(image)


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = np.array(image)


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_array = image_array / 255.0


    # --------------------------------------------------------
    # Transpose image
    # --------------------------------------------------------

    image_array = np.transpose(
        image_array
    )


    # --------------------------------------------------------
    # Add channel dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        image_array,
        verbose=0
    )


    predicted_index = np.argmax(
        prediction
    )


    predicted_character = labels[
        predicted_index
    ]


    confidence = (
        np.max(prediction) * 100
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("Prediction")

    st.success(
        f"Predicted Character: {predicted_character}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # ========================================================
    # TOP 5 PREDICTIONS
    # ========================================================

    st.subheader("Top Predictions")

    top_indices = np.argsort(
        prediction[0]
    )[-5:][::-1]


    for index in top_indices:

        character = labels[index]

        probability = (
            prediction[0][index] * 100
        )

        st.write(
            f"{character}: {probability:.2f}%"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    "Built using TensorFlow, CNN and the EMNIST Balanced dataset."
)