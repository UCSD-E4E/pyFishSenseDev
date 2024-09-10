from abc import ABC, abstractmethod
from typing import Tuple

import cv2
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

    def get_points_camera_space(
        self, camera_matrix: np.ndarray
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        empty_dist_coeffs = np.zeros((5,))
        ret, rotation, translation = cv2.solvePnP(
            self.get_points_body_space(),
            self.get_points_image_space(),
            camera_matrix,
            empty_dist_coeffs,
        )

        if ret:
            return rotation, translation
        else:
            return None, None
