import hashlib
import json
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import numpy as np

from pyfishsensedev.points_of_interest.points_of_interest_detector import (
    PointsOfInterestDetector,
)


class LabelStudioPointsOfInterestDetector(PointsOfInterestDetector):
    def __init__(self, image_path: Path, label_studio_json_path: Path) -> None:
        super().__init__()

        self.__image_path = image_path
        self.__label_studio_json_path = label_studio_json_path
        self.__hash = hashlib.md5(image_path.read_bytes()).hexdigest()

    def find_points_of_interest(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.__label_studio_json_path.exists():
            raise IOError

        with self.__label_studio_json_path.open("r") as f:
            label_studio = json.load(f)

        for item in label_studio:
            path_string: str = item["data"]["img"]

            if path_string.startswith("https://e4e-nas.ucsd.edu:6021"):
                url = urlparse(path_string)

                if Path(url.path).stem != self.__hash:
                    continue

                if len(item["annotations"]) == 0:
                    continue

                if len(item["annotations"][0]["result"]) == 0:
                    return None, None
            else:
                raise NotImplementedError

            result = item["annotations"][0]["result"]
            fork = None
            snout = None

            for label in result:
                original_width = float(label["original_width"])
                original_height = float(label["original_height"])

                label_name = label["value"]["keypointlabels"][0]

                x = int(label["value"]["x"] * original_width / 100.0)
                y = int(label["value"]["y"] * original_height / 100.0)
                if label_name == "Fork":
                    fork = np.array([x, y])
                elif label_name == "Snout":
                    snout = np.array([x, y])

            if fork is not None and snout is not None:
                return fork, snout

        print(f"Couldn't find image {self.__image_path}")
        return None, None
