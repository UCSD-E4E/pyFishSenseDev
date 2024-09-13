from pathlib import Path

import numpy as np
import pymupdf


class Pdf:
    DPI = 300

    def __init__(self, file_name: Path) -> None:
        with pymupdf.open(file_name.absolute().as_posix()) as doc:
            page: pymupdf.Page = doc.load_page(0)
            pixmap: pymupdf.Pixmap = page.get_pixmap(dpi=Pdf.DPI)
            bytes = np.frombuffer(pixmap.samples, dtype=np.uint8)

            self._image = bytes.reshape(pixmap.height, pixmap.width, pixmap.n)

    @property
    def image(self) -> np.ndarray:
        return self._image
