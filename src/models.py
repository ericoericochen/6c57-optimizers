import numpy as np

from tqdm import tqdm
from abc import ABC, abstractmethod
from src.optimizers import Optimizer, SGD


DEFAULT_LR = 3e-4


class Model(ABC):
    def __init__(self):
        self._fitted = False
        self._thetas = []
        self._biases = []
        self.losses = []

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def compute_loss(self, pred: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def compute_gradients(self, pred: np.ndarray, X: np.ndarray, y: np.ndarray):
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: Optimizer = None,
        epochs: int = 1000,
        batch_size: int = 8,
    ):
        """
        Params:
            - X: features matrix (B x M) N - number of samples, M - number of features
            - y: target vector (B x N)
            - optimizer: optimizer for gradient descent
            - epochs: number of iterations
        """
        assert not self._fitted, f"{__class__.__name__} is already fitted."

        if optimizer is None:
            optimizer = SGD(lr=DEFAULT_LR)

        self.X = X
        self.y = y

        bs = X.shape[0]
        M, N = X.shape[1], y.shape[1]
        self.theta = np.random.randn(N, M)
        self.bias = np.zeros(N)

        optimizer.init_parameters([self.theta, self.bias])

        with tqdm(total=epochs) as pbar:
            for i in tqdm(range(epochs)):
                # select mini batch
                idx = np.random.choice(bs, batch_size)
                X_batch = X[idx]
                y_batch = y[idx]

                pred = self.predict(X_batch)

                # calculate loss
                loss = self.compute_loss(pred, y_batch)
                self.losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())

                # calculate gradients
                dtheta, dbias = self.compute_gradients(pred, X_batch, y_batch)

                # optimizer step
                theta, bias = optimizer.step([dtheta, dbias])
                self.theta = theta
                self.bias = bias

                if self.theta.shape[1] == 1:
                    self._thetas.append(self.theta[0][0].item())
                    self._biases.append(self.bias[0].item())

                pbar.update(1)

        self._fitted = True
