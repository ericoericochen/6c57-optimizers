from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, lr: float = 3e-4):
        self.lr = lr
        self.parameters = None

    def init_parameters(self, parameters: list[np.ndarray]):
        self.parameters = parameters

    @abstractmethod
    def step(self, gradients: list[np.ndarray]):
        pass


class SGD(Optimizer):
    def step(self, gradients: list[np.ndarray]):
        for i, param in enumerate(self.parameters):
            self.parameters[i] = param - self.lr * gradients[i]

        return self.parameters


class SGDMomentum(Optimizer):
    pass


class Adagrad(Optimizer):
    pass


class RMSProp(Optimizer):
    pass


class Adam(Optimizer):
    pass


class AdamW(Optimizer):
    pass


class Shampoo(Optimizer):
    pass
