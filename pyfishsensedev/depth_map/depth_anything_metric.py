# depth_anything_metric.py
import cv2
import torch
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def load_depth_model(encoder="vitl", dataset="hypersim", device="cpu"):
    max_depth = 20 if dataset == "hypersim" else 80
    model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
    ckpt_path = f"checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model
