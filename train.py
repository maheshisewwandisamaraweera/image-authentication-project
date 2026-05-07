import os
import numpy as np

from sklearn.model_selection import train_test_split

from src.preprocess import preprocess_image
from src.model import build_model

# Dataset lists
data = []
labels = []

categories = ["actual", "edited", "ai_generated"]

# Load images
for label, category in enumerate(categories):

    folder = f"data/{category}"

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = preprocess_image(path)

        if img is not None:
            data.append(img)
            labels.append(label)

# Convert to arrays
X = np.array(data)
y = np.array(labels)

print("Dataset shape:", X.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Build model
model = build_model()

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Save model
model.save("model.h5")

print("Model saved successfully!")