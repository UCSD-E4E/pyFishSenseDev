from abc import ABC, abstractmethod


class Model(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
