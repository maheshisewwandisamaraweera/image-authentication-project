import streamlit as st
from PIL import Image
import numpy as np
import cv2
import base64


from src.prediction import predict_image

from src.evaluation import (
    evaluate_dataset,
    get_metrics
)


from src.visualization import (
    confusion_plot
)



st.set_page_config(
    page_title="Image Authentication",
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
    "Image Authentication Framework"
)



# =====================================================
# NAVIGATION
# =====================================================


menu = st.sidebar.selectbox(

    "Menu",

    [

        "Single Image Prediction",

        "Dataset Evaluation",

        "Performance"

    ]

)



# =====================================================
# SINGLE IMAGE PREDICTION
# =====================================================


if menu=="Single Image Prediction":


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



        st.dataframe(
            df
        )



        st.subheader(
            "Images"
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


                st.write(
                    "Actual:",
                    row["Actual Class"]
                )


                st.write(
                    "Prediction:",
                    row["Prediction"]
                )


                st.write(
                    "Confidence:",
                    f"{row['Confidence']}%"
                )



                if row["Result"]=="Correct":


                    st.success(
                        "Correct"
                    )


                else:


                    st.error(
                        "Wrong"
                    )



            st.divider()





# =====================================================
# PERFORMANCE
# =====================================================


else:


    if "df" not in st.session_state:


        st.warning(
            "Run evaluation first"
        )



    else:


        metrics=get_metrics(
            st.session_state["df"]
        )



        c1,c2,c3,c4=st.columns(4)



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



        st.pyplot(

            confusion_plot(
                metrics["cm"]
            )

        )



        st.text(
            metrics["report"]
        )