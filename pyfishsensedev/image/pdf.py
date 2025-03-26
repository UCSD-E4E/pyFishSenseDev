from pathlib import Path

import cv2
import numpy as np
import pymupdf

from pyfishsensedev.library.constants import INCH_TO_M


class Pdf:
    DPI = 300

    def __init__(self, file_name: Path, rotation_degree: float = None) -> None:
        with pymupdf.open(file_name.absolute().as_posix()) as doc:
            page: pymupdf.Page = doc.load_page(0)
            pixmap: pymupdf.Pixmap = page.get_pixmap(dpi=Pdf.DPI)
            bytes = np.frombuffer(pixmap.samples, dtype=np.uint8)

            self.__image = bytes.reshape(pixmap.height, pixmap.width, pixmap.n)
            self.__rotation_matrix = np.eye(2, dtype=float)

            if rotation_degree is not None:
                self.__rotation_matrix = cv2.getRotationMatrix2D(
                    (self.width / 2, self.height / 2), rotation_degree, 1
                )
                self.__image = cv2.warpAffine(
                    self.__image, self.__rotation_matrix, (self.width, self.height)
                )

    @property
    def image(self) -> np.ndarray:
        return self.__image

    @property
    def width(self) -> int:
        _, width = self.__image.shape

        return width

    @property
    def height(self) -> int:
        height, _ = self.__image.shape

        return height

    def get_physical_measurements(self, points: np.ndarray) -> np.ndarray:
        return (points / float(Pdf.DPI)) * INCH_TO_M
