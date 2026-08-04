import streamlit as st
from PIL import Image
import numpy as np
import cv2
import base64
import pandas as pd


from src.prediction import predict_image

from src.evaluation import (
    evaluate_dataset,
    get_metrics
)


from src.visualization import (
    confusion_plot,
    metrics_plot
)

def show_confidence_bar(prediction, confidence):

    if prediction == "Actual":
        color = "#00C853"      # Green

    elif prediction == "Edited":
        color = "#FF9800"      # Orange

    elif prediction == "AI Generated":
        color = "#F44336"      # Red

    else:
        color = "#808080"

    confidence = float(confidence)

    st.markdown(
        f"""
        <div style="
            width:100%;
            background:#E0E0E0;
            height:25px;
            border-radius:15px;
        ">
            <div style="
                width:{confidence}%;
                background:{color};
                height:25px;
                border-radius:15px;
                text-align:center;
                color:white;
                font-weight:bold;
                line-height:25px;
            ">
                {confidence:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(
    page_title="Module 1 - Detection & classification of synthetic or actual images in legal evidence",
    layout="wide"
)



# =====================================================
# BACKGROUND IMAGE + SIDEBAR COLOR
# =====================================================


def get_base64_image(image_path):

    with open(image_path, "rb") as img:

        return base64.b64encode(
            img.read()
        ).decode()



background = get_base64_image(
   r"C:\Users\Sewwandi\Desktop\image-authentication-project\src\assets\background_1.jpg"
)



st.markdown(
    f"""
    <style>


    /* MAIN BACKGROUND */

    .stApp {{

        background-image:
        url("data:image/jpg;base64,{background}");

        background-size: cover;

        background-position: center;

        background-attachment: fixed;

    }}



    /* MAIN CONTENT TRANSPARENT CARD */

    .block-container {{

        background-color:
        rgba(255,255,255,0.88);

        border-radius:20px;

        padding:2rem;

    }}




    /* SIDEBAR COLOR */

    section[data-testid="stSidebar"] {{

        background-color:#1B263B;

    }}




    section[data-testid="stSidebar"] label {{

        color:white;

    }}



    section[data-testid="stSidebar"] p {{

        color:white;

    }}



    /* SIDEBAR MENU BOX */

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{

        background-color:#415A77;

        color:white;

        border-radius:8px;

    }}



    section[data-testid="stSidebar"] div[data-baseweb="select"] span {{

        color:white;

    }}



    </style>

    """,

    unsafe_allow_html=True
)




# =====================================================
# TITLE
# =====================================================


st.title(
    "Module 1 - Detection & classification of synthetic or actual images in legal evidence"
)



# =====================================================
# NAVIGATION
# =====================================================


menu = st.sidebar.selectbox(

    "Menu",

    [
        "About the Research",

        "Single Image Prediction",

        "Dataset Evaluation",

        "Performance"

    ]

)

# =====================================================
# ABOUT THE RESEARCH
# =====================================================

if menu == "About the Research":

    st.header("📖 About the Research")

    st.markdown("""
    ### Title

    **Detection & Classification of Synthetic or Actual Images in Legal Evidence**

    This research focuses on developing an intelligent image authentication framework
    capable of distinguishing between real photographs, manipulated images, and
    AI-generated images that may be presented as legal evidence.
    """)

    st.divider()

    st.subheader("🎯 Research Problem")

    st.write("""
    Digital images have become an important source of evidence in criminal
    investigations, civil disputes, insurance claims, and social media.
    However, recent advances in image editing software and generative AI
    make it increasingly difficult to determine whether an image is authentic.

    Existing forensic tools mainly perform binary classification and often
    cannot distinguish between edited images and AI-generated images.
    """)

    st.divider()

    st.subheader("🔍 Research Gap")

    st.info("""
    • Most existing systems classify images only as Real or Fake.

    • They do not separately identify manipulated images and AI-generated images.

    • Existing tools provide very limited confidence information.

    • Most systems are designed for technical experts rather than legal professionals.

    • Few systems explain their predictions in a user-friendly manner.
    """)

    st.divider()

    st.subheader("💡 Proposed Solution")

    st.success("""
    The proposed framework classifies images into three categories:

    ✅ Actual Images

    🟠 Edited Images

    🔴 AI Generated Images

    The system also provides:

    • Prediction confidence score

    • Visual confidence indicator

    • Dataset evaluation

    • Performance metrics

    • Explainable results for legal evidence analysis
    """)

    st.divider()

    st.subheader("🎯 Objectives")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        **Main Objective**

        Develop an intelligent framework capable of detecting and classifying
        synthetic and actual images used as legal evidence.
        """)

    with col2:

        st.markdown("""
        **Specific Objectives**

        • Detect actual images

        • Detect edited images

        • Detect AI-generated images

        • Display confidence scores

        • Evaluate model performance
        """)

    st.divider()

    st.subheader("⚙️ Technologies Used")

    st.markdown("""
    - Python

    - TensorFlow / Keras

    - MobileNet

    - OpenCV

    - NumPy

    - Streamlit

    - Google Colab

    - VS Code
    """)

    st.divider()

    st.subheader("📊 Dataset")

    st.write("""
    The dataset consists of three image categories:

    🟢 Actual Images

    🟠 Edited Images

    🔴 AI Generated Images

    All images are preprocessed before training the MobileNet model.
    """)

# =====================================================
# SINGLE IMAGE PREDICTION
# =====================================================


elif menu=="Single Image Prediction":


    file = st.file_uploader(

        "Upload Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )



    if file:


        img_pil = Image.open(
            file
        )


        st.image(
            img_pil,
            width=400
        )



        img = np.array(
            img_pil
        )



        label,confidence = predict_image(
            img
        )



        if label=="Actual":

            st.success(label)


        elif label=="Edited":

            st.warning(label)


        else:

            st.error(label)




        st.progress(
            confidence
        )


        st.write(

            f"Confidence : {confidence*100:.2f}%"

        )





# =====================================================
# DATASET EVALUATION
# =====================================================


elif menu=="Dataset Evaluation":


    st.header(
        "567 Image Evaluation"
    )



    if st.button(
        "Start Evaluation"
    ):



        bar = st.progress(
            0
        )


        text = st.empty()



        def update(current,total):

            value = int(
                current/total*100
            )


            bar.progress(
                value
            )


            text.write(

                f"Processing {current}/{total}"

            )




        df = evaluate_dataset(
            update
        )



        st.session_state["df"] = df



        st.success(
            "Evaluation completed"
        )


        


        if "df" in st.session_state:

            df = st.session_state["df"]

            display_df = df.drop(
                columns=[
                   "Actual Class",
                   "Result"
              ]
            )

            st.subheader(
             "Images"
            )

            st.dataframe(
                display_df
            )


        for _,row in df.iterrows():



            col1,col2 = st.columns(
                [1,2]
            )



            with col1:


                img = cv2.imread(
                    row["Image Path"]
                )


                img=cv2.cvtColor(

                    img,

                    cv2.COLOR_BGR2RGB

                )


                st.image(
                    img,
                    width=220
                )




            with col2:


                st.write(
                    "Image Name:",
                    row["Image Name"]
                )


                # st.write(
                #     "Actual:",
                #     row["Actual Class"]
                # )


                st.write(
                    "Prediction:",
                    row["Prediction"]
                )


                st.write(
                    "Confidence:",
                    # f"{row['Confidence']}%"
                )

                show_confidence_bar(
                    row["Prediction"],
                    row["Confidence"]
                )



                # if row["Result"]=="Correct":


                #     st.success(
                #         "Correct"
                #     )


                # else:


                #     st.error(
                #         "Wrong"
                #     )



            st.divider()





# =====================================================
# PERFORMANCE
# =====================================================


elif menu=="Performance":


    if "df" not in st.session_state:


        st.warning(
            "Run evaluation first"
        )


    else:


        metrics = get_metrics(
            st.session_state["df"]
        )



        # =====================================================
        # PERFORMANCE CARDS
        # =====================================================


        c1,c2,c3,c4 = st.columns(4)



        c1.metric(
            "Accuracy",
            f"{metrics['accuracy']*100:.2f}%"
        )



        c2.metric(
            "Precision",
            f"{metrics['precision']*100:.2f}%"
        )



        c3.metric(
            "Recall",
            f"{metrics['recall']*100:.2f}%"
        )



        c4.metric(
            "F1 Score",
            f"{metrics['f1']*100:.2f}%"
        )



        # =====================================================
        # CONFUSION MATRIX
        # =====================================================


        st.subheader(
            "Confusion Matrix"
        )



        st.pyplot(

            confusion_plot(
                metrics["cm"]
            )

        )



        # =====================================================
        # CLASSIFICATION REPORT TABLE
        # =====================================================


        st.subheader(
            "Classification Report"
        )



        report_df = pd.DataFrame(
            metrics["report"]
        ).transpose()



        st.dataframe(
            report_df
        )



        # =====================================================
        # PRECISION RECALL F1 GRAPH
        # =====================================================


        st.subheader(
            "Precision, Recall and F1 Score Visualization"
        )



        st.pyplot(

            metrics_plot(
                metrics["report"]
            )

        )