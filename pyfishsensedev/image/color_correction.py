import numpy as np
from skimage.restoration import denoise_tv_chambolle, estimate_sigma

from pyfishsensedev.depth_map.depth_map import DepthMap
from pyfishsensedev.library.seathru import run_pipeline


class __Args:
    def __init__(self, f, l, p, min_depth, max_depth, spread_data_fraction):
        self.f = f
        self.l = l
        self.p = p
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.spread_data_fraction = spread_data_fraction

    def __iter__(self):
        return self.__dict__.__iter__()


class ColorCorrection:
    def __init__(
        self,
        f=2.0,
        l=0.5,
        p=0.01,
        min_depth=0.0,
        max_depth: float = None,
        spread_data_fraction=0.05,
    ):
        self.__f = f
        self.__l = l
        self.__min_depth = min_depth
        self.__max_depth = max_depth
        self.__spread_data_fraction = spread_data_fraction

    def correct_color(self, img: np.ndarray, depth_map: DepthMap) -> np.ndarray:
        args = __Args(
            self.__f,
            self.__l,
            self.__min_depth,
            self.__max_depth or depth_map.depth_map.max(),
            self.__spread_data_fraction,
        )
        recovered = run_pipeline(img, depth_map.depth_map, args)
        sigma_est = (
            estimate_sigma(recovered, channel_axis=2, average_sigmas=True) / 10.0
        )
        recovered = denoise_tv_chambolle(recovered, sigma_est, channel_axis=2)

        return recovered
