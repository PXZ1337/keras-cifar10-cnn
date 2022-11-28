import numpy as np

def schedule_fn(epoch: int) -> float:
    learning_rate = 1e-3
    if epoch < 5:
        learning_rate = 1e-3
    elif epoch < 20:
        learning_rate = 5e-4
    else:
        learning_rate = 1e-4
    return learning_rate


def schedule_fn2(epoch: int) -> float:
    if epoch < 10:
        return 1e-3
    else:
        return float(1e-3 * np.exp(0.1 * (10 - epoch)))


def schedule_fn3(epoch: int) -> float:
    return float(1e-3 * np.exp(0.1 * (10 - epoch)))


def schedule_fn4(epoch: int) -> float:
    return float(1e-3 * np.exp(0.05 * (10 - epoch)))
