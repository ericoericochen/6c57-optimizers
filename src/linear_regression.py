import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.optimizers import Optimizer, SGD


DEFAULT_LR = 3e-4


class LinearRegression:
    def __init__(self):
        self._fitted = False
        self._thetas = []
        self._biases = []
        self.losses = []

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
        if self._fitted:
            raise ValueError("LinearRegression model is already fitted.")

        if optimizer is None:
            optimizer = SGD(lr=DEFAULT_LR)

        self.X = X
        self.y = y

        bs = X.shape[0]
        M, N = X.shape[1], y.shape[1]
        self.theta = np.random.randn(M, N)
        self.bias = np.zeros(N)

        optimizer.init_parameters([self.theta, self.bias])

        with tqdm(total=epochs) as pbar:
            for i in tqdm(range(epochs)):
                # select mini batch
                idx = np.random.choice(bs, batch_size)
                X_batch = X[idx]
                y_batch = y[idx]

                pred = X_batch @ self.theta.T + self.bias

                # calculate loss
                loss = np.mean((pred - y_batch) ** 2)
                self.losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())

                # calculate gradients
                error = pred - y_batch
                dtheta = 2 / (N * M) * error.T @ X_batch
                dbias = 2 / (N * M) * error.sum(axis=0)

                # optimizer step
                theta, bias = optimizer.step([dtheta, dbias])
                self.theta = theta
                self.bias = bias

                if self.theta.shape[1] == 1:
                    self._thetas.append(self.theta[0][0].item())
                    self._biases.append(self.bias[0].item())

                pbar.update(1)

        self._fitted = True

    def predict(self, X: np.ndarray):
        return X @ self.theta.T + self.bias

    def display_contour_plot(self):
        if not self._fitted:
            raise ValueError("Model is not fitted yet.")

        # Ensure this works only for single-feature linear regression (2D plot)
        assert (
            self.theta.shape[1] == 1
        ), "Contour plot is only available for single-feature (2D) regression."

        # Generate a range of values for theta and bias
        theta_span = max(self._thetas) - min(self._thetas)
        bias_span = max(self._biases) - min(self._biases)
        theta_range = np.linspace(
            min(self._thetas) - 1, self._thetas[-1] + theta_span + 1, 100
        )
        bias_range = np.linspace(
            min(self._biases) - 1, self._biases[-1] + bias_span + 1, 100
        )
        theta_grid, bias_grid = np.meshgrid(theta_range, bias_range)

        # Compute the loss over the grid
        loss_grid = np.zeros_like(theta_grid)
        for i in range(theta_grid.shape[0]):
            for j in range(theta_grid.shape[1]):
                pred = self.X * theta_grid[i, j] + bias_grid[i, j]
                loss_grid[i, j] = np.mean((pred - self.y) ** 2)

        # Plot the contours
        plt.figure(figsize=(8, 6))
        cp = plt.contour(theta_grid, bias_grid, loss_grid, levels=30, cmap="gray")
        plt.colorbar(cp, label="Loss")

        # Plot the trajectory of the loss
        plt.plot(
            self._thetas,
            self._biases,
            "r-",
            marker="o",
            markersize=4,
            label="Loss trajectory",
        )

        plt.xlabel(r"$\theta$")
        plt.ylabel(r"Bias")
        plt.title("Loss Contour and Trajectory")
        plt.legend()
        plt.grid(True)
        plt.show()
