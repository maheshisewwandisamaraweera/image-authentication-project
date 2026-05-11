import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Image Authentication Framework",
    page_icon="🛡️",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f6fa;
}

section[data-testid="stSidebar"] {
    background-color: #1E145A;
}

.sidebar-title {
    color: white;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 30px;
}

.menu-box {
    background-color: rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: white;
}

.active-menu {
    background-color: #5B3FD0;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.label-text {
    font-size: 13px;
    font-weight: 600;
}

.big-text {
    font-size: 32px;
    font-weight: bold;
}

.small-text {
    font-size: 14px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = load_model("model.h5")

categories = [
    "ACTUAL IMAGE",
    "EDITED IMAGE",
    "AI-GENERATED IMAGE"
]

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🛡️ Image Authentication Framework</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="menu-box active-menu">📷 Module 1<br><small>Image Classification</small></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="menu-box">🛠️ Module 2<br><small>Image Restoration</small></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="menu-box">📄 Module 3<br><small>Image Explanation</small></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="menu-box">📊 Reports</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="menu-box">⚙️ Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="menu-box">ℹ️ About</div>',
        unsafe_allow_html=True
    )

# ------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------
def preprocess_image(image):

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image = cv2.resize(image, (224, 224))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("Module 1 – Image Classification")

st.write(
    "Detect and classify the image as Actual, Edited, or AI-generated with confidence scores."
)

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload New Image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------------------------
# MAIN CONTENT
# ------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1.1, 1])

    # --------------------------------------------
    # LEFT PANEL
    # --------------------------------------------
    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🖼️ Input Image")

        st.image(image, use_container_width=True)

        st.write(f"• {uploaded_file.name}")

        st.write("")

        st.write("Format:", uploaded_file.type)

        st.write(
            "Size:",
            f"{round(uploaded_file.size / (1024*1024), 2)} MB"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------
    # RIGHT PANEL
    # --------------------------------------------
    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📊 Classification Results")

        processed = preprocess_image(image)

        prediction = model.predict(processed)[0]

        predicted_index = np.argmax(prediction)

        predicted_class = categories[predicted_index]

        # RESULT COLORS
        if predicted_index == 0:

            bg = "#E9F8EE"
            color = "#16a34a"

            desc = "This image is classified as an actual (real) image."

        elif predicted_index == 1:

            bg = "#FFF4E8"
            color = "#f97316"

            desc = "This image is classified as an edited/manipulated image."

        else:

            bg = "#FDECEC"
            color = "#ef4444"

            desc = "This image is classified as an AI-generated image."

        # RESULT BOX
        st.markdown(
            f"""
            <div class="result-box" style="background-color:{bg};">
                <div class="label-text">
                    PREDICTED CLASS
                </div>

                <div class="big-text" style="color:{color};">
                    {predicted_class}
                </div>

                <div class="small-text">
                    {desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------
        # CONFIDENCE PANEL
        # ----------------------------------------
        st.markdown("### Confidence Score Panel")

        labels = [
            "ACTUAL (REAL)",
            "EDITED (MANIPULATED)",
            "AI-GENERATED (SYNTHETIC)"
        ]

        colors = [
            "#22c55e",
            "#f97316",
            "#ef4444"
        ]

        for i in range(3):

            score = float(prediction[i])

            st.markdown(
                f"""
                <div style="
                    color:{colors[i]};
                    font-weight:bold;
                    margin-bottom:5px;
                ">
                    {labels[i]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(score)

            st.markdown(
                f"""
                <div style="
                    text-align:right;
                    color:{colors[i]};
                    font-weight:bold;
                    margin-bottom:15px;
                ">
                    {score*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "The higher the score, the more confident the model is."
        )

        st.markdown('</div>', unsafe_allow_html=True)