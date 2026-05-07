from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_model():

    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    predictions = Dense(3, activation='softmax')(x)

    model = Model(
        inputs=base_model.input,
        outputs=predictions
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model