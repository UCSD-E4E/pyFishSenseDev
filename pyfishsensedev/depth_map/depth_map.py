from abc import ABC, abstractmethod

import numpy as np


class DepthMap(ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def depth_map(self) -> np.ndarray:
        raise NotImplementedError
