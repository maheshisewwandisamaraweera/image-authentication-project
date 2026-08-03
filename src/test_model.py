import cv2

from src.prediction import predict_image


img = cv2.imread(
    "test.jpg"
)


result, confidence = predict_image(img)


print(result)
print(confidence)