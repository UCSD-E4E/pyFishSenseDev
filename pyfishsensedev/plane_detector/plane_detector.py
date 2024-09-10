from abc import ABC, abstractmethod

import numpy as np



class PlaneDetector(ABC):
    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def width(self) -> int:
        _, width, _ = self._image.shape
        return width

    @property
    def height(self) -> int:
        height, _, _ = self._image.shape
        return height

    @abstractmethod
    def get_points_image_space(self) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def get_points_body_space(self) -> np.ndarray | None:
        raise NotImplementedError

    def is_valid(self) -> bool:
        return True
