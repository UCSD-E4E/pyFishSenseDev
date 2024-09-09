from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime

from pyfishsensedev.fish.fish_segmentation_fishial import FishSegmentationFishial


# Adapted from https://github.com/fishial/fish-identification/blob/main/module/segmentation_package/interpreter_segm.py
class FishSegmentationFishialOnnx(FishSegmentationFishial):
    def __init__(self) -> None:
        super().__init__()

        self.model_path = self.download_model().as_posix()
        self.ort_session = onnxruntime.InferenceSession(self.model_path)

    @property
    def _model_path(self) -> Path:
        return self._model_cache_path / "fishial.onnx"

    @property
    def _model_url(self) -> str:
        return "https://huggingface.co/ccrutchf/fishial/resolve/main/fishial.onnx?download=true"

    def unwarp_tensor(self, tensor: Tuple) -> Tuple:
        return tensor

    def inference(self, img: np.ndarray) -> np.ndarray:
        resized_img, scales = self._resize_img(img)

        ort_inputs = {
            self.ort_session.get_inputs()[0]
            .name: resized_img.astype("float32")
            .transpose(2, 0, 1)
        }
        ort_outs = self.ort_session.run(None, ort_inputs)

        complete_mask = self._convert_output_to_mask_and_polygons(ort_outs, scales, img)
        return complete_mask
