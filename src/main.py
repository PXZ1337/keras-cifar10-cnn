import os
import shutil

from cnn.Cifar10 import Cifar10
from cnn.model import build_model

from scipy.stats import randint, uniform
import numpy as np

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import ParameterSampler

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
EPOCHS = 20
RAND_SEARCH_N_TO_CALC = 10
BATCH_SIZE = 128
DATA = Cifar10()
NUM_CLASSES = 10

"""
    Initial try and testing out parameter selection.
"""


def train_initial_model():
    tb_callback = TensorBoard(log_dir=os.path.join(LOGS_DIR, 'model_initial'), histogram_freq=0, profile_batch=0)
    train_dataset = DATA.get_train_set()
    val_dataset = DATA.get_val_set()

    hyper_params = {
        "img_shape": DATA.img_shape,
        "num_classes": NUM_CLASSES,
        "learning_rate": 0.001,
        "conf_filter_1": 32,
        "kernel_size_1": 2,
        "conf_filter_2": 48,
        "kernel_size_2": 2,
        "conf_filter_3": 64,
        "kernel_size_3": 2,
        "dense_layer_size": 128,
        "activation": "relu",
        "optimizer": Adam
    }

    model = build_model(**hyper_params)
    model.fit(
        x=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[tb_callback],
        validation_data=val_dataset
    )


if __name__ == '__main__':
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    train_initial_model()
