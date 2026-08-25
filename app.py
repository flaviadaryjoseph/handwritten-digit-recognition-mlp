import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_mlp_model.keras")


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("✍️ Handwritten Digit Recognition")

st.write(
    "An Artificial Neural Network (MLP) trained on the "
    "MNIST dataset to recognize handwritten digits from 0 to 9."
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.info(
    """
    **Model:** Multi-Layer Perceptron (MLP)  
    **Dataset:** MNIST  
    **Input:** 28 × 28 grayscale image  
    **Classes:** 10 digits (0–9)  
    **Optimizer:** Adam
    """
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

with st.expander("🧠 View Model Architecture"):

    st.write("The neural network used in this project:")

    st.code(
        """
Input Layer
    ↓
784 neurons
    ↓
Dense Layer – 256 neurons (ReLU)
    ↓
Dropout – 20%
    ↓
Dense Layer – 128 neurons (ReLU)
    ↓
Output Layer – 10 neurons (Softmax)
    ↓
Predicted Digit (0–9)
        """
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload a Handwritten Digit")

uploaded_file = st.file_uploader(
    "Upload a PNG, JPG or JPEG image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Display original image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=250)

    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    image = ImageOps.grayscale(image)

    # Convert image to numpy array
    image_array = np.array(image)

    # Check background brightness
    # MNIST uses black background and white digit
    border_pixels = np.concatenate([
        image_array[0, :],
        image_array[-1, :],
        image_array[:, 0],
        image_array[:, -1]
    ])

    # If background is bright, invert the image
    if np.mean(border_pixels) > 127:
        image = ImageOps.invert(image)

    # Resize to MNIST dimensions
    image = image.resize((28, 28))

    # Convert to array
    processed_image = np.array(image)

    # Normalize
    processed_image = processed_image.astype("float32") / 255.0

    # Display processed image
    with col2:
        st.subheader("Processed Image")
        st.image(
            processed_image,
            width=250,
            clamp=True
        )

    # Flatten image
    input_data = processed_image.reshape(1, 784)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict(
        input_data,
        verbose=0
    )[0]

    predicted_digit = int(np.argmax(probabilities))

    confidence = float(np.max(probabilities)) * 100


    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    st.divider()

    st.subheader("🔮 Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Predicted Digit",
            value=str(predicted_digit)
        )

    with col2:

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )


    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    st.subheader("📊 Prediction Probabilities")

    probability_data = pd.DataFrame(
        {
            "Digit": list(range(10)),
            "Probability": probabilities * 100
        }
    )

    st.bar_chart(
        probability_data.set_index("Digit")
    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.divider()

    st.subheader("📈 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Input Features",
            "784"
        )

    with col2:
        st.metric(
            "Hidden Layers",
            "2"
        )

    with col3:
        st.metric(
            "Output Classes",
            "10"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Handwritten Digit Recognition using MLP/ANN | MNIST Dataset"
)