import tensorflow as tf
from typing import Any


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir: str, **kwargs: Any) -> None:
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        logs.update(
            {"learning_rate": self.model.optimizer.learning_rate.numpy()})
        super().on_epoch_end(epoch, logs)
