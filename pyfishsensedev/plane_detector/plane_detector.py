from abc import ABC, abstractmethod

import cv2
import numpy as np

from pyfishsensedev.calibration import LensCalibration


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

    def get_body_to_camera_space_transform(
        self, lens_calibration: LensCalibration
    ) -> np.ndarray | None:
        empty_dist_coeffs = np.zeros((5,))
        ret, rotation_vectors, translation = cv2.solvePnP(
            self.get_points_body_space(),
            self.get_points_image_space(),
            lens_calibration.camera_matrix,
            empty_dist_coeffs,
        )

        if not ret:
            return None

        rotation, _ = cv2.Rodrigues(rotation_vectors)

        translation = np.zeros((4, 4), dtype=float)
        translation[0:3, 0:3] = rotation
        translation[3, 0:3] = translation
        translation[3, 3] = 1

        return translation

    def get_points_camera_space(
        self, lens_calibration: LensCalibration
    ) -> np.ndarray | None:
        transformation = self.get_body_to_camera_space_transform(lens_calibration)
        body_points = self.get_points_body_space()

        if transformation is None or body_points is None:
            return None

        point_count, _ = body_points.shape
        homogeneous_body_points = np.zeros((point_count, 4), dtypes=float)
        homogeneous_body_points[:, :3] = body_points

        homogeneous_camera_points = transformation @ homogeneous_body_points
        return homogeneous_camera_points[:, :3]

    def get_normal_vector_camera_space(
        self, lens_calibration: LensCalibration
    ) -> np.ndarray | None:
        camera_points = self.get_points_camera_space(lens_calibration)

        if camera_points is None:
            return camera_points

        return np.cross(
            camera_points[1, :] - camera_points[0, :],
            camera_points[2, :] - camera_points[0, :],
        )

    def project_point_onto_plane_camera_space(
        self, point_image_space: np.ndarray, lens_calibration: LensCalibration
    ) -> np.ndarray | None:
        homogenous_point_image_space = np.zeros(3, dtype=float)
        homogenous_point_image_space[:2] = point_image_space
        homogenous_point_image_space[2] = 1

        # define laser ray assuming pinhole camera
        ray = lens_calibration.inverted_camera_matrix @ homogenous_point_image_space

        camera_points = self.get_points_camera_space(lens_calibration)
        normal_vector = self.get_normal_vector_camera_space(lens_calibration)

        if camera_points is None or normal_vector is None:
            return None

        # find scale factor such that the laser ray intersects with the plane
        scale_factor = (normal_vector.T @ camera_points[0, :]) / (normal_vector.T @ ray)
        return normal_vector * scale_factor
