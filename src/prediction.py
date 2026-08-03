import tensorflow as tf
import numpy as np
import cv2


MODEL_PATH = "models/model.h5"


model = tf.keras.models.load_model(
    MODEL_PATH
)


classes = [
    "Actual",
    "Edited",
    "AI Generated"
]


IMG_SIZE = 224



def preprocess_image(image):


    # Remove alpha channel
    if len(image.shape)==3:

        if image.shape[-1]==4:

            image=image[:,:,:3]


    image=cv2.resize(
        image,
        (IMG_SIZE,IMG_SIZE)
    )


    image=image.astype(
        "float32"
    )


    image=image/255.0


    image=np.expand_dims(
        image,
        axis=0
    )


    return image




def predict_image(image):


    processed=preprocess_image(
        image
    )


    prediction=model.predict(
        processed,
        verbose=0
    )


    index=np.argmax(
        prediction[0]
    )


    confidence=float(
        np.max(prediction[0])
    )


    label=classes[index]


    return label,confidence