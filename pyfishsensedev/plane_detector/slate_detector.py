import numpy as np

from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.library.homography.image_matcher import ImageMatcher
from pyfishsensedev.library.homography.utils import numpy_image_to_torch
from pyfishsensedev.plane_detector.plane_detector import PlaneDetector


class SlateDetector(PlaneDetector):
    def __init__(self, image: np.ndarray[np.uint8], pdf: Pdf) -> None:
        super().__init__(image)

        self._pdf = pdf
        self._image_matcher = ImageMatcher(
            numpy_image_to_torch(self._pdf.image),
            processing_conf={
                "preprocess": {"gamma": 2.0, "sharpness": None, "scale": 1.6},
                "extractor": {"max_num_keypoints": None},
                "matcher": {"filter_threshold": 0.5},
            },
        )

    def _get_template_matches(self):
        feats0_matches, feats1_matches = self._image_matcher(numpy_image_to_torch(self.image))

        print("Done")

    def _get_points_image_space(self):
        pass

    def _get_points_body_space(self):
        pass
