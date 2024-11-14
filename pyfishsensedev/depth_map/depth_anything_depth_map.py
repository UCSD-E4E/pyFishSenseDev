import numpy as np
from PIL import Image
from transformers import pipeline

from pyfishsensedev.depth_map.depth_map import DepthMap


class DepthAnythingDepthMap(DepthMap):
    @property
    def depth_map(self) -> np.ndarray:
        return self.__depth_map

    def __init__(self, img: np.ndarray, device: str, scale=0.8) -> None:
        super().__init__()

        pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=device,
        )
        image = Image.fromarray(img)
        self.__depth_map = 255.0 - np.array(pipe(image)["depth"], dtype=float) * scale

    def rescale(self, scale: float):
        self.__depth_map *= scale
