from pathlib import Path
from typing import Iterator, Self, Tuple

import numpy as np

from pyfishsensedev.calibration.lens_calibration import LensCalibration
from pyfishsensedev.laser.laser_detector import LaserDetector
from pyfishsensedev.library.array_read_write import (
    read_laser_calibration,
    write_laser_calibration,
)
from pyfishsensedev.library.laser_parallax import atanasov_calibration_method
from pyfishsensedev.plane_detector.plane_detector import PlaneDetector


class LaserCalibration:
    def __init__(
        self,
        laser_axis: np.ndarray | None = None,
        laser_position: np.ndarray | None = None,
    ) -> None:
        self._laser_axis = laser_axis
        self._laser_position = laser_position

        either_not_none = laser_axis is not None or laser_position is not None
        both_not_none = laser_axis is not None and laser_position is not None

        if either_not_none and not both_not_none:
            raise ValueError(
                "Either laser_axis or laser_position are none.  Please either set both to be none or set both values."
            )

    @property
    def laser_axis(self) -> np.ndarray:
        return self._laser_axis

    @property
    def laser_position(self) -> np.ndarray:
        return self._laser_position

    def load(self, calibration_path: Path):
        self._laser_position, self._laser_axis = read_laser_calibration(
            calibration_path.absolute().as_posix()
        )

    def save(self, calibration_path: Path):
        if self.laser_position is None or self.laser_axis is None:
            raise ValueError(
                "laser_position or laser_axis are None.  Please call load or calibrate and try again."
            )

        write_laser_calibration(
            calibration_path.absolute().as_posix(),
            self.laser_axis,
            self.laser_position,
        )

    def plane_calibrate(
        self,
        calibration_planes_and_dark_images: Iterator[Tuple[PlaneDetector, np.ndarray]],
        lens_calibration: LensCalibration,
        laser_calibration_estimate: Self,
        device: str,
    ) -> None:
        calibration_planes_and_dark_images = [
            (p, d) for p, d in calibration_planes_and_dark_images if p.is_valid()
        ]

        laser_detector = LaserDetector(
            lens_calibration, laser_calibration_estimate, device
        )

        laser_points = [
            (p, laser_detector.find_laser(d))
            for p, d in calibration_planes_and_dark_images
        ]

        laser_points_3d = [
            p.project_point_onto_plane_camera_space(l)
            for p, l in laser_points
            if l is not None
        ]
        laser_points = [l for l in laser_points if l is not None]

        laser_params = atanasov_calibration_method(laser_points_3d)

        self.laser_axis = laser_params[:3]

        self.laser_position = np.zeros(3, dtype=float)
        self.laser_position[:2] = laser_params[:-2]
