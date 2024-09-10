import numpy as np

from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.depth_map.depth_map import DepthMap


class LaserDepthMap(DepthMap):
    def __init__(
        self,
        laser_image_position: np.ndarray,
        lens_calibration: LensCalibration,
        laser_calibration: LaserCalibration,
    ) -> None:
        super().__init__()

        self._laser_image_position = laser_image_position
        self._lens_calibration = lens_calibration
        self._laser_calibration = laser_calibration

    @property
    def depth_map(self) -> np.ndarray:
        pass
