import streamlit as st
import numpy as np
import cv2

from tensorflow.keras.models import load_model

# Load model
model = load_model("model.h5")

categories = ["Actual", "Edited", "AI Generated"]

# Preprocess uploaded image
def preprocess(img):

    img = cv2.resize(img, (224,224))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# UI
st.title("Image Intelligent Authentication Framework")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Convert uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR")

    # Prediction
    processed = preprocess(img)

    prediction = model.predict(processed)[0]

    predicted_index = np.argmax(prediction)

    st.subheader(
        f"Prediction: {categories[predicted_index]}"
    )

    # Confidence scores
    colors = ["🟢", "🟠", "🔴"]

    for i in range(3):

        st.write(
            f"{colors[i]} {categories[i]}: {prediction[i]*100:.2f}%"
        )

        st.progress(float(prediction[i]))