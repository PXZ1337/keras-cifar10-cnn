from typing import Tuple

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Activation, Dense, Flatten, MaxPool2D, Input, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


def build_model(
    img_shape: Tuple[int, int, int],
    num_classes: int,
    learning_rate: float,
    conf_filter_1: int,
    kernel_size_1: int,
    conf_filter_2: int,
    kernel_size_2: int,
    conf_filter_3: int,
    kernel_size_3: int,
    dense_layer_size: int,
    activation: str,
    optimizer: tf.keras.optimizers.Optimizer,
    use_batch_normalization: bool = False,
    use_global_max_pooling: bool = False
) -> Model:
    input_img = Input(shape=img_shape)

    """ Conv-Block-1"""
    x = Conv2D(filters=conf_filter_1, kernel_size=kernel_size_1, padding="same")(input_img)
    x = Activation(activation)(x)
    x = Conv2D(filters=conf_filter_1, kernel_size=kernel_size_1, padding="same")(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D()(x)

    """ Conv-Block-2"""
    x = Conv2D(filters=conf_filter_2, kernel_size=kernel_size_2, padding="same")(x)
    x = Activation(activation)(x)
    x = Conv2D(filters=conf_filter_2, kernel_size=kernel_size_2, padding="same")(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D()(x)

    """ Conv-Block-3"""
    x = Conv2D(filters=conf_filter_3, kernel_size=kernel_size_3, padding="same")(x)
    x = Activation(activation)(x)
    x = Conv2D(filters=conf_filter_3, kernel_size=kernel_size_3, padding="same")(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D()(x)

    """
        Convert a 4D Tensor to a Vector for the Dense-Layer by using either a flatten layer
        or an max pooling layer which should remove less important data from the feature map.
    """
    if use_global_max_pooling:
        x = GlobalMaxPooling2D()(x)
    else:
        x = Flatten()(x)
    x = Dense(units=dense_layer_size)(x)
    if use_batch_normalization:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    """ One-Hot output"""
    x = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[input_img], outputs=[y_pred])

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer(learning_rate), metrics=["accuracy"]
    )

    return model
