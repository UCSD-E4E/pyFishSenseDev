from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np


class ImageProcessor(ABC):
    def __init__(self, file: Path) -> None:
        super().__init__()

        self.file = file

    def __iter__(self) -> Self:
        return self

    @abstractmethod
    def __next__(self) -> np.array:
        raise NotImplementedError
