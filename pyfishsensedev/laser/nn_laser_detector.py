from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import peak_local_max

from pyfishsensedev.calibration.laser_calibration import LaserCalibration
from pyfishsensedev.calibration.lens_calibration import LensCalibration
from pyfishsensedev.laser.laser_detector import LaserDetector
from pyfishsensedev.library.online_ml_model import OnlineMLModel


class LaserDetectorNetwork(nn.Module):
    def __init__(self):
        super(LaserDetectorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3200, 10)
        self.act2 = nn.ReLU()

        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.flatten(x)
        x = self.act2(self.linear1(x))

        x = self.linear2(x)

        return x


class NNLaserDetector(LaserDetector, OnlineMLModel):
    @property
    def name(self) -> str:
        return "NNLaserDetector"

    def __init__(
        self,
        lens_calibration: LensCalibration,
        laser_calibration: LaserCalibration,
        device: str,
    ):
        super().__init__()

        model_weights_path = self.download_model()

        self.lens_calibration = lens_calibration
        self.laser_calibration = laser_calibration
        self.model = LaserDetectorNetwork()
        self.model.load_state_dict(
            torch.load(
                model_weights_path.as_posix(), map_location=device, weights_only=True
            )
        )

        self.model.to(device)
        self.device = device

    @property
    def _model_path(self) -> Path:
        return self._model_cache_path / "laser_detector.pth"

    @property
    def _model_url(self) -> str:
        return "https://huggingface.co/ccrutchf/laser-detector/resolve/main/laser_detection.pth"

    def _get_2d_from_3d(self, vector: np.ndarray) -> np.ndarray:
        homogeneous_coords = vector / vector[2]
        return self.lens_calibration.camera_matrix @ homogeneous_coords

    def _get_line(
        self, one: np.ndarray, two: np.ndarray
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        slope = (two[1] - one[1]) / (two[0] - one[0])

        x1 = 0
        y1 = int(slope * (-one[0]) + one[1])

        if y1 < 0:
            y1 = 0
            x1 = int((-one[1]) / slope + one[0])

        # We can now calculate the point of convergence.

        x2, y2 = two

        return (x1, y1), (x2, y2)

    def _return_line(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        first_point = (
            self.laser_calibration.laser_position
            + 1 * self.laser_calibration.laser_axis
        )
        second_point = (
            self.laser_calibration.laser_position
            + 10000 * self.laser_calibration.laser_axis
        )

        first_2d = self._get_2d_from_3d(first_point)
        second_2d = self._get_2d_from_3d(second_point)

        first_2d_tup = (int(first_2d[0]), int(first_2d[1]))
        second_2d_tup = (int(second_2d[0]), int(second_2d[1]))

        return self._get_line(first_2d_tup, second_2d_tup)

    def _get_masked_image_matrix(
        self, img: np.ndarray, thickness=75
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        # Make sure that the file is a JPG
        if type(img) != np.ndarray:
            return None, None

        if len(np.shape(img)) == 3:
            # Get the line
            img_clone = img.copy()
            start_point, end_point = self._return_line()
            cv2.line(
                img_clone,
                start_point,
                end_point,
                color=(0, 0, 255),
                thickness=thickness,
            )

            # Create the mask and the masked image
            mask = cv2.inRange(img_clone, (0, 0, 255), (0, 0, 256)) / 255
            mask_rgb = np.dstack((mask, mask, mask)).astype(np.uint8)
            masked_image = img * mask_rgb

            return masked_image, mask

        elif len(np.shape(img)) == 2:
            img_clone = img.copy()
            start_point, end_point = self._return_line()
            cv2.line(img_clone, start_point, end_point, color=0, thickness=thickness)

            mask = cv2.inRange(img_clone, 0, 1) / 255
            mask = mask.astype(np.uint8)
            masked_image = img * mask

            return masked_image, mask

    def _get_tiles(
        self, img: np.ndarray, coordinates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # takes in gray-scale image
        # get 20 by 20 tiles that are centred around the points in coordinates
        N = 20
        M = 20
        buf = []
        coords_buf = []
        img_tiles = [
            img[int(x - M / 2) : int(x + M / 2), int(y - N / 2) : int(y + N / 2), :]
            for x, y in coordinates
        ]
        for i, tile in enumerate(img_tiles):
            if tile.shape == (N, M, 3):
                buf.append(np.reshape(tile, (3, N, M)))
                coords_buf.append(coordinates[i])
        coordinates = np.asarray(coords_buf)
        img_tiles = np.asarray(buf)

        return img_tiles, coordinates

    def _correct_laser_dot(self, coord: np.ndarray, img: np.ndarray) -> np.ndarray:
        red = img[:, :, 2]
        x, y = coord

        laser_mask = np.zeros_like(red)
        laser_mask[y - 43 : y + 43, x - 43 : x + 43] = 255

        mask = np.zeros_like(red)
        mask[np.logical_and(245 <= red, laser_mask == 255)] = 255
        mask = cv2.UMat(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        c = max(contours, key=cv2.contourArea)

        M = cv2.moments(c)

        if M["m00"] == 0:
            return None

        cX = float(M["m10"] / M["m00"])
        cY = float(M["m01"] / M["m00"])

        return np.array([cX, cY])

    def find_laser(self, img: np.ndarray, thickness=75) -> np.ndarray | None:
        masked_img, _ = self._get_masked_image_matrix(img, thickness=thickness)

        if masked_img is None:
            return None

        coordinates = peak_local_max(img[:, :, 2], min_distance=20)

        tiles, coords = self._get_tiles(img, coordinates)
        self.model.eval()

        if len(tiles) == 0:
            return None

        preds = self.model(torch.Tensor(tiles).to(self.device))
        possible = (F.sigmoid(preds) > 0.99).cpu()
        possible_coords = []

        for i in range(coords.shape[0]):
            if possible[i]:
                possible_coords.append(coords[i, :])

        possible_coords = np.array(possible_coords)

        if len(possible_coords) == 0:
            return None

        allowed_coords = []
        final_coord = []
        _, vanishing_point = np.array(self._return_line())

        for i in range(possible_coords.shape[0]):
            coord = possible_coords[i, :]

            if np.any(masked_img[coord[0], coord[1]]):
                allowed_coords.append([coord[0], coord[1]])

        if len(allowed_coords) == 0:
            return None

        min_distance = 1000000000
        for coord in allowed_coords:
            distance = np.linalg.norm(coord - vanishing_point)
            if distance < min_distance:
                final_coord = coord
                min_distance = distance

        uncorrected = np.array([final_coord[1], final_coord[0]])
        corrected = self._correct_laser_dot(uncorrected, img)  # was previously y, x

        if corrected is None:
            return uncorrected

        return corrected


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from pyfishsensedev.image import ImageRectifier, RawProcessor

    raw_processor = RawProcessor()
    raw_processor_dark = RawProcessor(enable_histogram_equalization=False)
    image_rectifier = ImageRectifier(Path("./data/lens-calibration.pkg"))
    laser_detector = LaserDetector(
        Path("./data/models/laser_detection.pth"),
        Path("./data/lens-calibration.pkg"),
        Path("./data/laser-calibration.pkg"),
    )

    img = raw_processor.load_and_process(Path("./data/P8030201.ORF"))
    img_dark = raw_processor_dark.load_and_process(Path("./data/P8030201.ORF"))
    img = image_rectifier.rectify(img)
    img_dark = image_rectifier.rectify(img_dark)

    img8 = ((img.astype("float64") / 65535) * 255).astype("uint8")
    img_dark8 = ((img_dark.astype("float64") / 65535) * 255).astype("uint8")
    coords = laser_detector.find_laser(img_dark8)

    plt.imshow(img8)
    plt.plot(coords[0], coords[1], "r.")
    plt.show()
