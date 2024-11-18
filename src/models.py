from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        self._fitted = False

    @abstractmethod
    def _fit(self):
        pass

    def fit(self):
        assert not self._fitted, f"{__class__.__name__} is already fitted."
        return self._fit()
