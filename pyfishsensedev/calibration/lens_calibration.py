from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from pyfishsensedev.library.array_read_write import (
    read_camera_calibration,
    write_camera_calibration,
)
from pyfishsensedev.plane_detector.plane_detector import PlaneDetector


class LensCalibration:
    def __init__(self) -> None:
        self._camera_matrix: np.ndarray = None
        self._distortion_coefficients: np.ndarray = None

    @property
    def camera_matrix(self) -> np.ndarray:
        return self._camera_matrix

    @property
    def inverted_camera_matrix(self) -> np.ndarray:
        return np.invert(self.camera_matrix)

    @property
    def distortion_coefficients(self) -> np.ndarray:
        return self._distortion_coefficients

    def load(self, calibration_path: Path):
        self._camera_matrix, self._distortion_coefficients = read_camera_calibration(
            calibration_path.absolute().as_posix()
        )

    def save(self, calibration_path: Path):
        write_camera_calibration(
            calibration_path.absolute().as_posix(),
            self._camera_matrix,
            self._distortion_coefficients,
        )

    def calibrate(
        self, calibration_planes: Iterator[PlaneDetector], max_error=None
    ) -> float:
        calibration_planes = [p for p in calibration_planes if p.is_valid()]

        height, width = calibration_planes[0].width, calibration_planes[0].height

        body_space = [p.get_points_body_space() for p in calibration_planes]
        image_space = [p.get_points_image_space() for p in calibration_planes]

        error, self._camera_matrix, self._distortion_coefficients, _, _ = (
            cv2.calibrateCamera(body_space, image_space, (width, height), None, None)
        )

        if max_error is not None and error > max_error:
            raise ValueError(
                f"While calibrating, an error of {error} was above the maximum allowed error of {max_error}."
            )

        return error
