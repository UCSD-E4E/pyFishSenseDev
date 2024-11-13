from pathlib import Path
from typing import Self

import numpy as np
import rawpy
import skimage

from pyfishsensedev.image.image_processors.image_processor import ImageProcessor


class RawProcessor(ImageProcessor):
    def __init__(self, file: Path, gamma=0.3) -> None:
        super().__init__(file)

        self.__gamma = gamma
        self.__has_iterated = True

    def __iter__(self) -> Self:
        self.__has_iterated = False

        return self

    def __next__(self) -> np.ndarray:
        if self.__has_iterated:
            raise StopIteration

        self.__has_iterated = True

        with rawpy.imread(self.file.as_posix()) as raw:
            img = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)
            img = skimage.exposure.adjust_gamma(img, gamma=self.__gamma)
            img = skimage.exposure.equalize_adapthist(img)

            return self.__double_2_uint16(img)

    def __double_2_uint16(self, img: np.ndarray) -> np.ndarray:
        return (img * 65535).astype(np.uint16)
