import os
import shutil

from cnn.Cifar10 import Cifar10
from cnn.model import build_model
from cnn.lr_scheduler_fn import schedule_fn, schedule_fn2, schedule_fn3, schedule_fn4
from cnn.tf_callback import LRTensorBoard

from scipy.stats import randint, uniform
import numpy as np

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import ParameterSampler

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
EPOCHS = 20
RAND_SEARCH_N_TO_CALC = 10
BATCH_SIZE = 128
DATA = Cifar10()
NUM_CLASSES = 10

BEST_PARAMS = {
    "img_shape": DATA.img_shape,
    "num_classes": NUM_CLASSES,
    "learning_rate": 1e-3,
    "conf_filter_1": 32,
    "kernel_size_1": 3,
    "conf_filter_2": 64,
    "kernel_size_2": 3,
    "conf_filter_3": 128,
    "kernel_size_3": 3,
    "dense_layer_size": 128,
    "activation": "relu",
    "optimizer": Adam
}

"""
    Determine good working combinations of parameters for this spicific problem.
"""


def train_models_with_random_search():
    train_dataset = DATA.get_train_set()
    val_dataset = DATA.get_val_set()

    # Paramas from which we want to sample
    param_distribution = {
        "learning_rate": uniform(0.001, 0.0001),
        "conf_filter_1": randint(16, 64),
        "kernel_size_1": randint(2, 7),
        "conf_filter_2": randint(16, 64),
        "kernel_size_2": randint(2, 7),
        "conf_filter_3": randint(16, 64),
        "kernel_size_3": randint(2, 7),
        "dense_layer_size": randint(128, 1024)
    }
    results = {
        "best_score": -np.inf,
        "best_params": {},
        "val_scores": [],
        "params": [],
    }

    dist = ParameterSampler(param_distribution, n_iter=RAND_SEARCH_N_TO_CALC)

    print(f"Parameter combinations in total: {len(dist)}")

    for idx, comb in enumerate(dist):
        print(f"Running Comb: {idx}")

        model = build_model(
            img_shape=DATA.img_shape,
            num_classes=NUM_CLASSES,
            **comb,
            activation="relu",
            optimizer=Adam
        )

        model_log_dir = os.path.join(LOGS_DIR, f"model_rand_{idx}")

        if os.path.exists(model_log_dir):
            shutil.rmtree(model_log_dir)
            os.mkdir(model_log_dir)

        tb_callback = TensorBoard(log_dir=model_log_dir, histogram_freq=0, profile_batch=0)
        model.fit(
            train_dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_data=val_dataset,
            callbacks=[tb_callback],
        )

        val_accuracy = model.evaluate(val_dataset, batch_size=BATCH_SIZE, verbose=0)[1]
        results["val_scores"].append(val_accuracy)
        results["params"].append(comb)

    best_run_idx = np.argmax(results["val_scores"])
    results["best_score"] = results["val_scores"][best_run_idx]
    results["best_params"] = results["params"][best_run_idx]

    print(f"Best score: {results['best_score']} using params: {results['best_params']}\n")

    scores = results["val_scores"]
    params = results["params"]

    for idx, (score, param) in enumerate(zip(scores, params)):
        print(f"Idx: {idx} - Score: {score} with param: {param}")


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


"""
    Add Batch-Normalization to the best paramter set.
"""


def train_with_batch_normalization():
    tb_callback = TensorBoard(log_dir=os.path.join(LOGS_DIR, 'model_with_batch_normalization'),
                              histogram_freq=0, profile_batch=0)
    train_dataset = DATA.get_train_set()
    val_dataset = DATA.get_val_set()

    model = build_model(**BEST_PARAMS, use_batch_normalization=True)
    model.fit(
        x=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[tb_callback],
        validation_data=val_dataset
    )


"""
    Try out different scheduler functions which reduce the learning rate while training the model.
"""


def train_with_lr_scheduler():
    train_dataset = DATA.get_train_set()
    val_dataset = DATA.get_val_set()
    schedule_functions = [schedule_fn, schedule_fn2, schedule_fn3, schedule_fn4]

    for schedule in schedule_functions:
        lrs_callback = LearningRateScheduler(schedule=schedule, verbose=1)
        lr_callback = LRTensorBoard(log_dir=os.path.join(LOGS_DIR, f"model_with_scheduler_{schedule.__name__}"),
                                    histogram_freq=0, profile_batch=0)

        model = build_model(**BEST_PARAMS, use_batch_normalization=False)
        model.fit(
            x=train_dataset,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[lrs_callback, lr_callback],
            validation_data=val_dataset
        )


"""
    Combine all results and make some last adjustments
"""


def train_final_model():
    tb_callback = TensorBoard(log_dir=os.path.join(LOGS_DIR, 'model_final'),
                              histogram_freq=0, profile_batch=0)
    es_callback = EarlyStopping("val_accuracy",
                                patience=20,
                                verbose=1,
                                restore_best_weights=True,
                                min_delta=5e-5)

    plateau_callback = ReduceLROnPlateau(monitor="val_accuracy",
                                         factor=0.99,
                                         patience=3,
                                         verbose=1,
                                         min_lr=1e-5)

    train_dataset = DATA.get_train_set()
    val_dataset = DATA.get_val_set()

    model = build_model(**BEST_PARAMS, use_batch_normalization=True)
    model.fit(
        x=train_dataset,
        batch_size=BATCH_SIZE,
        epochs=100,
        callbacks=[tb_callback, es_callback, plateau_callback],
        validation_data=val_dataset
    )


if __name__ == '__main__':
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    # train_initial_model()
    # train_with_batch_normalization()
    # train_final_model()
    train_with_lr_scheduler()
