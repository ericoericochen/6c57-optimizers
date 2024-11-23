import numpy as np

from .optimizers import Optimizer
from .models import Model


class LogisticRegression(Model):
    def __init__(self):
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: Optimizer = None,
        epochs: int = 1000,
        batch_size: int = 8,
    ):
        pass
