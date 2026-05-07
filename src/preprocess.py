import cv2
import numpy as np

def preprocess_image(path):
    img = cv2.imread(path)

    if img is None:
        return None

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0

    return img