from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, lr: float = 3e-4):
        self.lr = lr
        self.parameters = None
        self.initialized = False

    def init_parameters(self, parameters: list[np.ndarray]):
        self.parameters = parameters
        self.initialized = True

    def step(self, gradients: list[np.ndarray]):
        assert self.initialized, "Parameters are not initialized."
        return self._step(gradients)

    @abstractmethod
    def _step(self, gradients: list[np.ndarray]):
        pass


class SGD(Optimizer):
    def _step(self, gradients: list[np.ndarray]):
        for i, param in enumerate(self.parameters):
            self.parameters[i] = param - self.lr * gradients[i]

        return self.parameters


class SGDMomentum(Optimizer):
    pass


class Adagrad(Optimizer):
    """
    Adagrad Optimizer from https://stanford.edu/~jduchi/projects/DuchiHaSi10_colt.pdf.

    It accumulates squared gradients from previous iterations and adapts the learning rate for each parameter.
    Parameters that have received large gradients will have a smaller update step and those with smaller gradients will
    have a larger update step.

    An advantage of Adagrad is that there is no need to tune the learning rate.
    """

    def __init__(self, lr: float = 1.0, eps: float = 1e-8):
        super().__init__(lr)
        self.eps = eps
        self.r = None  # squared gradients

    def init_parameters(self, parameters: list[np.ndarray]):
        super().init_parameters(parameters)
        self.r = [np.zeros_like(parameter) for parameter in parameters]

    def _step(self, gradients: list[np.ndarray]):
        # accumulate squared gradients
        for i, gradient in enumerate(gradients):
            self.r[i] += gradient**2

        # adapt learning rate
        lrs = [self.lr / (self.eps + np.sqrt(self.r[i])) for i in range(len(self.r))]

        for i, param in enumerate(self.parameters):
            self.parameters[i] = param - lrs[i] * gradients[i]

        return self.parameters


class RMSProp(Optimizer):
    pass


class Adam(Optimizer):
    pass


class AdamW(Optimizer):
    pass


class Shampoo(Optimizer):
    pass
