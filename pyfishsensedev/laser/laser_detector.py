from abc import ABC, abstractmethod

import numpy as np


class LaserDetector(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def find_laser(self, img: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError
