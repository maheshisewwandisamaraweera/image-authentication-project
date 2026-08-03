import os
import cv2
import pandas as pd


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


from src.prediction import predict_image



folders={

    "Actual":
    "data/actual",


    "Edited":
    "data/edited",


    "AI Generated":
    "data/ai_generated"

}




def evaluate_dataset(progress_callback=None):


    results=[]


    total=0


    for folder in folders.values():

        total += len(
            os.listdir(folder)
        )


    current=0



    for actual_class,folder in folders.items():


        images=os.listdir(folder)



        for image_name in images:


            image_path=os.path.join(
                folder,
                image_name
            )


            image=cv2.imread(
                image_path
            )


            prediction,confidence=predict_image(
                image
            )



            results.append({

                "Image Name":
                image_name,


                "Image Path":
                image_path,


                "Actual Class":
                actual_class,


                "Prediction":
                prediction,


                "Confidence":
                round(
                    confidence*100,
                    2
                ),


                "Result":

                "Correct"
                if actual_class==prediction

                else

                "Wrong"

            })



            current=current+1



            if progress_callback:


                progress_callback(
                    current,
                    total
                )



    df=pd.DataFrame(
        results
    )


    return df





def get_metrics(df):


    accuracy=accuracy_score(

        df["Actual Class"],

        df["Prediction"]

    )



    precision=precision_score(

        df["Actual Class"],

        df["Prediction"],

        average="weighted",

        zero_division=0

    )



    recall=recall_score(

        df["Actual Class"],

        df["Prediction"],

        average="weighted",

        zero_division=0

    )



    f1=f1_score(

        df["Actual Class"],

        df["Prediction"],

        average="weighted",

        zero_division=0

    )



    cm=confusion_matrix(

        df["Actual Class"],

        df["Prediction"]

    )



    report=classification_report(

        df["Actual Class"],

        df["Prediction"],

        zero_division=0

    )


    return {

        "accuracy":accuracy,

        "precision":precision,

        "recall":recall,

        "f1":f1,

        "cm":cm,

        "report":report

    }