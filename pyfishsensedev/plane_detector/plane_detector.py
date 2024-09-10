from abc import ABC, abstractmethod

import numpy as np


class PlaneDetector(ABC):
    def __init__(self, image: np.ndarray) -> None:
        self._image = image
        self._height, self._width, _ = image.shape
        self._points_image_space: np.ndarray | None = None
        self._points_body_space: np.ndarray | None = None

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @abstractmethod
    def _get_points_image_space(self) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def _get_points_body_space(self) -> np.ndarray | None:
        raise NotImplementedError

    @property
    def points_image_space(self) -> np.ndarray | None:
        if self._points_image_space is None:
            self._points_image_space = self._get_points_image_space()

        return self._points_image_space

    @property
    def points_body_space(self) -> np.ndarray | None:
        if self._points_body_space is None:
            self._points_body_space = self._get_points_body_space()

        return self._points_body_space

    def detect(self):
        # Force the points to be cached.
        _ = self.points_image_space
        _ = self.points_image_space

    def is_valid(self) -> bool:
        return (
            self.points_image_space is not None and self.points_body_space is not None
        )
