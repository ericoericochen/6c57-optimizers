import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .optimizers import Optimizer, SGD
from .models import Model


DEFAULT_LR = 3e-4


class LinearRegression(Model):
    def compute_loss(self, pred: np.ndarray, y: np.ndarray):
        loss = np.mean((pred - y) ** 2)
        return loss

    def predict(self, X: np.ndarray):
        return X @ self.theta.T + self.bias

    def compute_gradients(self, pred: np.ndarray, X: np.ndarray, y: np.ndarray):
        M, N = X.shape[0], y.shape[1]
        error = pred - y
        dtheta = 2 / (N * M) * error.T @ X
        dbias = 2 / (N * M) * error.sum(axis=0)

        return dtheta, dbias

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
