from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image
from requests import get

from pyfishsensedev.library.online_ml_model import OnlineMLModel
from pyfishsensedev.segmentation.segmentation import Segmentation


# Adapted from https://github.com/fishial/fish-identification/blob/main/module/segmentation_package/interpreter_segm.py
class FishSegmentationFishial(Segmentation, OnlineMLModel, ABC):
    MIN_SIZE_TEST = 800
    MAX_SIZE_TEST = 1333

    SCORE_THRESHOLD = 0.3
    MASK_THRESHOLD = 0.5

    @abstractmethod
    def unwarp_tensor(self, tensor: Iterable) -> Tuple:
        raise NotImplementedError

    def _resize_img(
        self, img: np.ndarray, interp_method=Image.LANCZOS
    ) -> Tuple[np.ndarray, np.ndarray]:
        height, width, _ = img.shape
        size = self.MIN_SIZE_TEST * 1.0
        scale = size / min(height, width)

        if height < width:
            new_height, new_width = size, scale * width
        else:
            new_height, new_height = scale * height, size

        if max(new_height, new_width) > self.MAX_SIZE_TEST:
            scale = self.MAX_SIZE_TEST * 1.0 / max(new_height, new_width)
            new_height *= scale
            new_width *= scale

        new_height += 0.5
        new_width += 0.5
        new_height = int(new_height)
        new_width = int(new_width)

        # pil_image = Image.fromarray(img)
        # pil_image = pil_image.resize((new_width, new_height), interp_method)

        # resized_img = np.asarray(pil_image)

        resized_img = cv2.resize(img, (new_width, new_height))

        # get scales of x&y after scaling
        scales = np.divide(img.shape[:2], resized_img.shape[:2])

        return resized_img, scales

    def _convert_output_to_mask_and_polygons(
        self,
        mask_rcnn_output: Tuple,
        scales: np.ndarray,
        img: np.ndarray,
    ) -> np.ndarray:

        def rescale_polygon_to_src_size(
            poly, start_pont: np.ndarray, scales: np.ndarray
        ) -> List[Tuple[int, int]]:
            return [
                (
                    int((start_pont[0] + point[0]) * scales[0]),
                    int((start_pont[1] + point[1]) * scales[1]),
                )
                for point in poly
            ]

        def do_paste_mask(masks: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
            # import torch
            # from torch.nn import functional as F

            """
            Args:
                masks: N, 1, H, W
                boxes: N, 4
                img_h, img_w (int):
                skip_empty (bool): only paste masks within the region that
                    tightly bound all boxes, and returns the results this region only.
                    An important optimization for CPU.

            Returns:
                if skip_empty == False, a mask of shape (N, img_h, img_w)
                if skip_empty == True, a mask of shape (N, h', w'), and the slice
                    object for the corresponding region.
            """
            masks = masks.squeeze()

            resized_mask: np.ndarray = cv2.resize(masks, (img_w, img_h))
            return resized_mask

            # masks = torch.tensor(masks)
            # x0_int, y0_int = 0, 0
            # x1_int, y1_int = img_w, img_h
            # x0, y0, x1, y1 = (
            #     np.array([[0]]),
            #     np.array([[0]]),
            #     np.array([[img_w]]),
            #     np.array([[img_h]]),
            # )

            # N = masks.shape[0]

            # img_y = torch.arange(y0_int, y1_int, dtype=torch.float32) + 0.5
            # img_x = torch.arange(x0_int, x1_int, dtype=torch.float32) + 0.5
            # img_y = (img_y - y0) / (y1 - y0) * 2 - 1
            # img_x = (img_x - x0) / (x1 - x0) * 2 - 1
            # # img_x, img_y have shapes (N, w), (N, h)
            # gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
            # gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
            # grid = torch.stack([gx, gy], dim=3)

            # resized_mask = F.grid_sample(
            #     masks, grid.to(masks.dtype), align_corners=False
            # )

            # return resized_mask

        def bitmap_to_polygon(bitmap):
            """Convert masks from the form of bitmaps to polygons.

            Args:
                bitmap (ndarray): masks in bitmap representation.

            Return:
                list[ndarray]: the converted mask in polygon representation.
                bool: whether the mask has holes.
            """
            bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
            # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
            #   into a two-level hierarchy. At the top level, there are external
            #   boundaries of the components. At the second level, there are
            #   boundaries of the holes. If there is another contour inside a hole
            #   of a connected component, it is still put at the top level.
            # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
            outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours = outs[-2]
            hierarchy = outs[-1]
            if hierarchy is None:
                return [], False
            # hierarchy[i]: 4 elements, for the indexes of next, previous,
            # parent, or nested contours. If there is no corresponding contour,
            # it will be -1.
            contours = [c.reshape(-1, 2) for c in contours]
            return sorted(contours, key=len, reverse=True)

        boxes, classes, masks, scores, img_size = self.unwarp_tensor(mask_rcnn_output)
        complete_mask = np.zeros_like(img, dtype=np.uint8)

        for ind in range(len(masks)):
            if scores[ind] <= self.SCORE_THRESHOLD:
                continue
            x1, y1, x2, y2 = (
                int(boxes[ind][0]),
                int(boxes[ind][1]),
                int(boxes[ind][2]),
                int(boxes[ind][3]),
            )
            mask_h, mask_w = y2 - y1, x2 - x1

            np_mask = do_paste_mask(masks[ind, None, :, :], mask_h, mask_w)

            # Threshold the mask converting to uint8 casuse opencv diesn't allow other type!
            np_mask = np.where(np_mask > self.MASK_THRESHOLD, 255, 0).astype(np.uint8)

            # Find contours in the binary mask
            contours = bitmap_to_polygon(np_mask)

            # Ignore empty contpurs and small artifacts
            if len(contours) < 1 or len(contours[0]) < 10:
                continue

            # Convert local polygon to src image
            polygon_full = rescale_polygon_to_src_size(contours[0], (x1, y1), scales)

            complete_mask = cv2.fillPoly(
                complete_mask, [np.array(polygon_full)], (ind + 1, ind + 1, ind + 1)
            )

        return complete_mask[:, :, 0]
