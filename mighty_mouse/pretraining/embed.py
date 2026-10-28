"""Per-frame CLS-token extraction from videos with a frozen pretrained ViT.

torch / transformers are imported inside the functions that need them, so importing
this module (and testing the numpy-only helpers) doesn't require them.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PATCH_SIZE = 16
DEFAULT_RESOLUTION = 224
DEFAULT_BATCH_SIZE = 256

MODEL_TYPES = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
}

# LVD-1689M web-image normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def output_path(embeddings_dir: Path, model_type: str, dataset: str, stem: str) -> Path:
    """`<embeddings_dir>/<model_type>/<dataset>/<stem>.npy`"""
    return Path(embeddings_dir) / model_type / dataset / f"{stem}.npy"


def preprocess_frames(frames: np.ndarray, resolution: int) -> np.ndarray:
    """Resize RGB frames to a square (no crop) and apply ImageNet normalization.

    Args:
        frames: uint8 RGB frames, shape (B, H, W, 3)
        resolution: target square side length; must be a multiple of the patch size (16)

    Returns:
        float32 array, shape (B, 3, resolution, resolution)
    """
    if resolution % PATCH_SIZE != 0:
        raise ValueError(f"resolution must be a multiple of {PATCH_SIZE}, got {resolution}")
    resized = np.stack([
        cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_CUBIC) for f in frames
    ])
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(0, 3, 1, 2).astype(np.float32)


def iter_frame_batches(video_file: Path, batch_size: int) -> Iterator[np.ndarray]:
    """Sequentially decode a video, yielding uint8 RGB batches of shape (<=B, H, W, 3).

    Raises:
        OSError: if the video can't be opened
    """
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise OSError(f"Error opening video file {video_file}")
    try:
        batch = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(batch) == batch_size:
                yield np.stack(batch)
                batch = []
        if batch:
            yield np.stack(batch)
    finally:
        cap.release()


def load_cls_embedder(model_type: str, device: str):
    """Load a frozen pretrained ViT in eval mode on `device`.

    Raises:
        ValueError: if `model_type` isn't one of `MODEL_TYPES`
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model_type {model_type!r}; choices: {list(MODEL_TYPES)}")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_TYPES[model_type])
    return model.eval().to(device)


def embed_video(
    model,
    video_file: Path,
    resolution: int = DEFAULT_RESOLUTION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_frames: int | None = None,
) -> np.ndarray:
    """Run `model` on every frame of a video and keep each frame's CLS token.

    Args:
        model: from `load_cls_embedder`
        video_file: path to the video
        resolution: square input side length, in pixels
        batch_size: frames per forward pass
        n_frames: expected frame count, only used to size the progress bar

    Returns:
        float16 array, shape (n_frames_decoded, hidden_size)
    """
    import torch

    device = next(model.parameters()).device
    out = []
    with tqdm(total=n_frames, desc=video_file.name, leave=False) as pbar:
        for frames in iter_frame_batches(video_file, batch_size):
            x = torch.from_numpy(preprocess_frames(frames, resolution)).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                cls = model(pixel_values=x).last_hidden_state[:, 0]
            out.append(cls.float().cpu().numpy().astype(np.float16))
            pbar.update(len(frames))
    if not out:
        raise OSError(f"No frames decoded from {video_file}")
    return np.concatenate(out)
