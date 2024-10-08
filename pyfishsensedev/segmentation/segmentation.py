from abc import ABC, abstractmethod

import numpy as np


class Segmentation(ABC):
    @abstractmethod
    def inference(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError
