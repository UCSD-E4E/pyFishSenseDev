from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely import ops
from shapely.plotting import plot_line, plot_points, plot_polygon

from pyfishsensedev.points_of_interest.fish.fish_geometry import FishGeometry
from pyfishsensedev.points_of_interest.general.pca_points_of_interest_detector import (
    PcaPointsOfInterestDetector,
)
from pyfishsensedev.points_of_interest.points_of_interest_detector import (
    PointsOfInterestDetector,
)


class FishPointsOfInterestDetector(PointsOfInterestDetector):
    """Fish mask endpoint detector and classifier.

    1. Estimates endpoints.
    2. Classifies endpoints as either head/tail. Uses endpoints from:
        - previous step OR
        - specified parameter
    3. Uses separate heuristic to correct each.

    Included is a function that does all these steps in one.
    """

    def __init__(self) -> None:
        super().__init__()

        self.geo: FishGeometry = None

    def correct_tail_coord(self):  # TODO
        tail_poly = self.geo.get_tail_poly()

        tail_convex = tail_poly.convex_hull

        tail_convex_diff = shapely.difference(tail_convex, tail_poly).geoms

        distances = [
            shapely.distance(p, self.geo.get_tailpoint_extended())
            for p in tail_convex_diff
        ]

        tail_convex_diff = tail_convex_diff[distances.index(min(distances))]

        tail_point = shapely.geometry.Point(self.geo.get_tail_coord())
        tailpoint_in_poly = tail_convex_diff.boundary.contains(
            tail_point
        ) or tail_point.within(tail_convex_diff)
        print("Tail in poly? " + str(tailpoint_in_poly))

        if tailpoint_in_poly:
            _, tail_corrected = ops.nearest_points(
                self.geo.get_headpoint_extended(), tail_convex_diff.boundary
            )
        else:
            # we assume that the tail is non-convex
            # TODO: slice tail_poly with a line a little before tailpoint_line, trim the extra, and get the centroid
            tail_corrected = shapely.geometry.Point(self.geo.get_tail_coord())

        # plot_polygon(tail_convex_diff, add_points=False)
        # plot_points(tail_corrected)

        self.geo.set_tail_corrected(np.asarray([tail_corrected.x, tail_corrected.y]))

        return {"tail": self.geo.get_tail_corrected()}

    def correct_head_coord(self):
        # any point on head_poly_sliced is better or just as good as the initial estimated point
        head_poly_sliced = ops.split(
            self.geo.get_head_poly(), self.geo.get_headpoint_line()
        ).geoms

        # polygon closest to the extended headpoint is the one we want
        distances = [
            shapely.distance(p, self.geo.get_headpoint_extended())
            for p in head_poly_sliced
        ]
        head_poly_sliced = head_poly_sliced[distances.index(min(distances))]

        # we guess the tip of head_poly_sliced by finding its nearest point to another point far ahead
        try:
            _, head_corrected = ops.nearest_points(
                self.geo.get_headpoint_extended(), head_poly_sliced.convex_hull.boundary
            )
        except:
            # if a point can't be extracted, we default to the original estimation
            head_corrected = shapely.geometry.Point(self.geo.get_head_coord())
        self.geo.set_head_corrected([head_corrected.x, head_corrected.y])

        return {"head": self.geo.get_head_corrected()}

    def correct_endpoints(self):
        self.correct_head_coord()

        self.correct_tail_coord()

        return {
            "head": self.geo.get_head_corrected(),
            "tail": self.geo.get_tail_corrected(),
        }

    def classify_endpoints(self, endpoints=None) -> Dict[str, np.ndarray]:
        endpoints = (
            self.geo.get_estimated_endpoints() if endpoints == None else endpoints
        )
        self.geo.set_estimated_endpoints(endpoints)

        # get halves
        halves = self.geo.get_halves()

        # get the convex versions
        halves_convex = self.geo.get_halves_convex()

        # get the differences
        halves_difference = [
            shapely.difference(halves_convex[0], halves[0]),
            shapely.difference(halves_convex[1], halves[1]),
        ]

        # compare the areas and set head/tail polys
        half_areas = [halves_difference[0].area], [halves_difference[1].area]
        self.geo.set_tail_poly(halves[half_areas.index(max(half_areas))])
        self.geo.set_head_poly(halves[half_areas.index(min(half_areas))])

        # assign the classification to the endpoints
        head_coord = self.geo.get_head_coord(endpoints)
        tail_coord = self.geo.get_tail_coord(endpoints)

        # compute the confidence score
        confidence = float(
            (max(half_areas)[0] - min(half_areas)[0]) / max(half_areas)[0] / 2.0 + 0.5
        )
        confidence = int(confidence * 100) / 100

        return {"head": head_coord, "tail": tail_coord, "confidence": confidence}

    def estimate_endpoints(self) -> Tuple[np.ndarray, np.ndarray, float]:
        mask = self.geo.get_mask()

        left_coord, right_coord = PcaPointsOfInterestDetector().find_points_of_interest(
            mask
        )

        self.geo.set_estimated_endpoints([left_coord, right_coord])
        return self.geo.get_estimated_endpoints()

    def find_points_of_interest(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.geo = FishGeometry(mask)

        self.estimate_endpoints()
        self.classify_endpoints()
        corrected = self.correct_endpoints()
        return corrected["head"], corrected["tail"]


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import torch

    from pyfishsensedev.image import ImageRectifier
    from pyfishsensedev.image.image_processors.raw_processor import RawProcessor
    from pyfishsensedev.laser.laser_detector import LaserDetector
    from pyfishsensedev.segmentation.fish import FishSegmentationFishialPyTorch

    raw_processor = RawProcessor()
    raw_processor_dark = RawProcessor(enable_histogram_equalization=False)
    image_rectifier = ImageRectifier(Path(".../data/calibration/fsl-01d-lens-raw.pkg"))
    laser_detector = LaserDetector(
        Path("../laser/models/laser_detection.pth"),
        Path(".../data/calibration/fsl-01d-lens-raw.pkg"),
        Path(".../data/calibration/fsl-01d-laser.pkg"),
    )

    img = raw_processor.load_and_process(Path(".../data/P8030201.ORF"))
    img_dark = raw_processor_dark.load_and_process(Path("./data/P8030201.ORF"))
    img = image_rectifier.rectify(img)
    img_dark = image_rectifier.rectify(img_dark)

    img8 = ((img.astype("float64") / 65535) * 255).astype("uint8")
    img_dark8 = ((img_dark.astype("float64") / 65535) * 255).astype("uint8")
    coords = laser_detector.find_laser(img_dark8)

    fish_segmentation_inference = FishSegmentationFishialPyTorch(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    segmentations = fish_segmentation_inference.inference(img8)

    mask = np.zeros_like(segmentations, dtype=bool)
    mask[segmentations == segmentations[coords[1], coords[0]]] = True

    fish_head_tail_detector = FishPointsOfInterestDetector(mask)

    # run through the process
    estimations = fish_head_tail_detector.estimate_endpoints()
    fish_head_tail_detector.classify_endpoints(estimations)
    corrections = fish_head_tail_detector.correct_endpoints()
    head_coord = corrections["head"]
    tail_coord = corrections["tail"]

    # or just use one function
    # corrections = fish_head_tail_detector.get_head_tail()
    # head_coord = corrections['head']
    # tail_coord = corrections['tail']

    plt.imshow(img8)
    plt.plot(head_coord[0], head_coord[1], "r.")
    plt.plot(tail_coord[0], tail_coord[1], "b.")
    plt.show()
