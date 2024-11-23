import numpy as np
from .models import Model


def softmax(z: np.ndarray):
    e_z = np.exp(z)
    Z = np.sum(e_z, axis=1, keepdims=True)
    s = e_z / Z

    return s


class LogisticRegression(Model):
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
