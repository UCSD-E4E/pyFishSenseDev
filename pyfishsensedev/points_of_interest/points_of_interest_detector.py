from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class PointsOfInterestDetector(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def find_points_of_interest(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
