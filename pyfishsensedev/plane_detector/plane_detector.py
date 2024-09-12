from abc import ABC, abstractmethod

import cv2
import numpy as np

from pyfishsensedev.library.laser_parallax import image_coordinate_to_projected_point


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

    def get_body_to_camera_space_transform(
        self, camera_matrix: np.ndarray
    ) -> np.ndarray | None:
        empty_dist_coeffs = np.zeros((5,))
        ret, rotation_vectors, translation = cv2.solvePnP(
            self.points_body_space,
            self.points_image_space,
            camera_matrix,
            empty_dist_coeffs,
        )

        if not ret:
            return None

        rotation, _ = cv2.Rodrigues(rotation_vectors)

        transformation = np.zeros((4, 4), dtype=float)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation.flatten()
        transformation[3, 3] = 1

        return transformation

    def get_points_camera_space(
        self,
        camera_matrix: np.ndarray,
    ) -> np.ndarray | None:
        transformation = self.get_body_to_camera_space_transform(camera_matrix)
        body_points = self.points_body_space

        if transformation is None or body_points is None:
            return None

        point_count, _ = body_points.shape
        homogeneous_body_points = np.zeros((point_count, 4), dtype=float)
        homogeneous_body_points[:, :3] = body_points
        homogeneous_body_points[:, 3] = 1

        homogeneous_camera_points = np.einsum(
            "ij,kj->ki", transformation, homogeneous_body_points
        )
        return homogeneous_camera_points[:, :3]

    def get_normal_vector_camera_space(
        self,
        camera_matrix: np.ndarray,
    ) -> np.ndarray | None:
        camera_points = self.get_points_camera_space(camera_matrix)

        if camera_points is None:
            return camera_points

        return np.cross(
            camera_points[1, :] - camera_points[0, :],
            camera_points[2, :] - camera_points[0, :],
        )

    def project_point_onto_plane_camera_space(
        self,
        point_image_space: np.ndarray,
        camera_matrix: np.ndarray,
        inverted_camera_matrix: np.ndarray,
    ) -> np.ndarray | None:
        ray = image_coordinate_to_projected_point(
            point_image_space, inverted_camera_matrix
        )

        camera_points = self.get_points_camera_space(camera_matrix)
        normal_vector = self.get_normal_vector_camera_space(
            camera_matrix
        ) / np.linalg.norm(normal_vector)

        if camera_points is None or normal_vector is None:
            return None

        # find scale factor such that the laser ray intersects with the plane
        scale_factor = (normal_vector.T @ camera_points[0, :]) / (normal_vector.T @ ray)
        return normal_vector * scale_factor

    def is_valid(self) -> bool:
        return (
            self.points_image_space is not None and self.points_body_space is not None
        )
