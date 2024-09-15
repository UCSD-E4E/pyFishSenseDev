import numpy as np

from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.library.homography.image_matcher import ImageMatcher
from pyfishsensedev.library.homography.utils import numpy_image_to_torch
from pyfishsensedev.plane_detector.plane_detector import PlaneDetector


class SlateDetector(PlaneDetector):
    def __init__(self, image: np.ndarray[np.uint8], pdf: Pdf) -> None:
        super().__init__(image)

        self._pdf = pdf
        self._tensor_template = numpy_image_to_torch(self._pdf.image)

    def _get_template_matches(self):
        max_num_keypoints = 0
        max_feats0_matches = None
        max_feats1_matches = None

        for scale in range(1, 20):
            scale = float(scale) / 10.0

            image_matcher = ImageMatcher(
                self._tensor_template,
                processing_conf={
                    "preprocess": {"gamma": 2.0, "sharpness": None, "scale": scale},
                    "extractor": {"max_num_keypoints": None},
                    "matcher": {"filter_threshold": 0.5},
                },
            )

            feats0_matches, feats1_matches = image_matcher(
                numpy_image_to_torch(self.image)
            )

            num_keypoints, _ = feats0_matches.shape
            if num_keypoints > max_num_keypoints:
                max_num_keypoints = num_keypoints

                max_feats0_matches = feats0_matches
                max_feats1_matches = feats1_matches

        print(f"{max_feats0_matches.size()}, {max_feats1_matches.size()}")

    def _get_points_image_space(self):
        pass

    def _get_points_body_space(self):
        pass
