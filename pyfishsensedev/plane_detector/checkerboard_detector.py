import cv2
import numpy as np

from pyfishsensedev.plane_detector.projectable_plane_detector import (
    ProjectablePlaneDetector,
)


class CheckerboardDetector(ProjectablePlaneDetector):
    def __init__(
        self, image: np.ndarray[np.uint8], rows: int, columns: int, square_size: float
    ) -> None:
        super().__init__(image)

        self._rows = rows
        self._columns = columns
        self._square_size = square_size

    def _get_points_image_space(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(
            gray, (self._rows, self._columns), None
        )

        if ret:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)

            return corners
        else:
            return None

    def _get_points_body_space(self):
        # coordinates of squares in the checkerboard world space
        objp = np.zeros((self._rows * self._columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self._rows, 0 : self._columns].T.reshape(-1, 2)
        objp = self._square_size * objp

        return objp
