from typing import Tuple

import numpy as np
import torch

from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.library.homography.image_matcher import ImageMatcher
from pyfishsensedev.library.homography.utils import numpy_image_to_torch
from pyfishsensedev.plane_detector.plane_detector import PlaneDetector


class SlateDetector(PlaneDetector):
    def __init__(self, image: np.ndarray[np.uint8], pdf: Pdf) -> None:
        super().__init__(image)

        self._pdf = pdf
        self._tensor_template = numpy_image_to_torch(self._pdf.image)
        self._ran_template_matches = False
        self._feats0_matches = None
        self._feats1_matches = None

    def _get_template_matches(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._ran_template_matches:
            max_num_keypoints = 0

            for scale in range(1, 20):
                scale = float(scale) / 10.0

                image_matcher = ImageMatcher(
                    self._tensor_template,
                    com_license=False,
                    processing_conf={
                        "preprocess": {"gamma": 2.0, "sharpness": None, "scale": scale},
                        "matcher": {"filter_threshold": 0.5},
                    },
                )

                feats0_matches, feats1_matches = image_matcher(
                    numpy_image_to_torch(self.image)
                )

                num_keypoints, _ = feats0_matches.shape
                if num_keypoints > max_num_keypoints:
                    max_num_keypoints = num_keypoints

                    self._max_feats0_matches = feats0_matches
                    self._max_feats1_matches = feats1_matches

            self._ran_template_matches = True

        return self._max_feats0_matches, self._feats1_matches

    def _get_points_image_space(self):
        pass

    def _get_points_body_space(self):
        pass

    def is_valid(self):
        feats0_matches, _ = self._get_template_matches()

        num_matches, _ = feats0_matches.shape

        return num_matches > 10
