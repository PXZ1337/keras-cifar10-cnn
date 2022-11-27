import os

from typing import Tuple
from Cifar10 import Cifar10

from tensorflow.keras.layers import Conv2D, Activation, Dense, Flatten, MaxPool2D, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')


def build_model(img_shape: Tuple[int, int, int], num_classes: int) -> Model:
    input_img = Input(shape=img_shape)

    x = Conv2D(filters=32, kernel_size=2, padding="same")(input_img)
    x = Activation("relu")(x)
    x = Conv2D(filters=32, kernel_size=2, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=48, kernel_size=2, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=48, kernel_size=2, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=64, kernel_size=2, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=2, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(units=128)(x)
    x = Activation("relu")(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[input_img], outputs=[y_pred])

    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"]
    )

    return model


if __name__ == '__main__':
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    epochs = 20
    batch_size = 128
    data = Cifar10()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    tb_callback = TensorBoard(log_dir=os.path.join(LOGS_DIR, 'model_initial'), histogram_freq=0, profile_batch=0)
    model = build_model(img_shape=data.img_shape, num_classes=10)
    model.fit(
        x=train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tb_callback],
        validation_data=val_dataset
    )
