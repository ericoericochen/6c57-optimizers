import numpy as np
from .models import Model


def softmax(z: np.ndarray):
    e_z = np.exp(z)
    Z = np.sum(e_z, axis=1, keepdims=True)
    s = e_z / Z

    return s


class LogisticRegression(Model):
    @staticmethod
    def one_hot_encode(labels: np.ndarray) -> np.ndarray:
        """
        Converts a 1D array of labels into a 2D one-hot encoded array.

        Args:
            labels (np.ndarray): Input array of shape (N,), where N is the number of samples.

        Returns:
            np.ndarray: One-hot encoded array of shape (N, C),
                        where C is the number of unique classes.
        """
        if labels.ndim != 1:
            raise ValueError("Input array must be 1-dimensional.")

        # Determine the number of unique classes
        num_classes = np.max(labels) + 1

        # Create the one-hot encoded array
        one_hot = np.eye(num_classes)[labels]

        return one_hot

    def compute_loss(self, pred: np.ndarray, y: np.ndarray):
        target_probs = pred * y
        target_probs = target_probs[target_probs != 0]
        cross_entropy_loss = -np.log(target_probs).mean()

        return cross_entropy_loss

    def predict(self, X: np.ndarray):
        logits = X @ self.theta.T + self.bias
        probs = softmax(logits)

        return probs

    def compute_gradients(self, pred: np.ndarray, X: np.ndarray, y: np.ndarray):
        B = pred.shape[0]
        error = pred - y
        dtheta = (1 / B) * error.T @ X
        dbias = error.mean(axis=0)

        return dtheta, dbias
