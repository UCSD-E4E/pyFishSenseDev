from abc import ABC, abstractmethod

import numpy as np


class DepthMap(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_depth_map(self) -> np.ndarray:
        raise NotImplementedError
