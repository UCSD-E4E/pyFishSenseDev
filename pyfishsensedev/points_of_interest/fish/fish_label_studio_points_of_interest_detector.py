import json
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import numpy as np

from pyfishsensedev.points_of_interest.points_of_interest_detector import (
    PointsOfInterestDetector,
)


class FishLabelStudioPointsOfInterestDetector(PointsOfInterestDetector):
    def __init__(self, image_path: Path, label_studio_json_path: Path) -> None:
        super().__init__()

        self.__image_path = image_path
        self.__label_studio_json_path = label_studio_json_path

    def find_points_of_interest(self, _: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.__label_studio_json_path.exists():
            raise IOError

        with self.__label_studio_json_path.open("r") as f:
            label_studio = json.load(f)

        for item in label_studio:
            path_string: str = item["data"]["img"]

            if path_string.startswith("https://e4e-nas.ucsd.edu:6021"):
                url = urlparse(path_string)

                if Path(url.path).stem != self.__image_path.stem:
                    continue

                if len(item["annotations"]) == 0:
                    continue

                if len(item["annotations"][0]["result"]) == 0:
                    return None

                result_array = item["annotations"][0]["result"]

                head: np.ndarray | None = None
                tail: np.ndarray | None = None

                for result in result_array:
                    if result["type"] != "keypointlabels":
                        continue

                    label = result["value"]["keypointlabels"][0]
                    original_width = float(result["original_width"])
                    original_height = float(result["original_height"])

                    x = result["value"]["x"] * original_width / 100.0
                    y = result["value"]["y"] * original_height / 100.0

                    array = np.array([x, y])

                    if label == "Snout":
                        head = array
                    elif label == "Fork":
                        tail = array

                if head is None or tail is None:
                    return None

                return head, tail

            else:
                raise NotImplementedError

        raise KeyError("laser label cannot be found.")
