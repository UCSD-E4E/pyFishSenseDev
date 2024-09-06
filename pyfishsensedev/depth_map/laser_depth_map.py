import numpy as np

from pyfishsensedev.depth_map.depth_map import DepthMap


class LaserDepthMap(DepthMap):
    def __init__(self, laser_image_position: np.ndarray) -> None:
        super().__init__()

        self._laser_image_position = laser_image_position

    @property
    def laser_image_position(self):
        return self._laser_image_position
