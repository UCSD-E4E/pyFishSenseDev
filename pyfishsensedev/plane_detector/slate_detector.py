import numpy as np

from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.plane_detector.plane_detector import PlaneDetector


class SlateDetector(PlaneDetector):
    def __init__(self, image: np.ndarray[np.uint8], pdf: Pdf) -> None:
        super().__init__(image)

        self._pdf = pdf

    def _get_points_image_space(self):
        pass

    def _get_points_body_space(self):
        pass
