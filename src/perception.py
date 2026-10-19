"""Convert rendered frames into policy observations."""

from __future__ import annotations

import cv2
import numpy as np

# Canonical observation geometry shared by the RL environment wrappers.
FRAME_WIDTH = 84
FRAME_HEIGHT = 84


def preprocess_frame(
    frame: np.ndarray,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> np.ndarray:
    """Convert RGB or grayscale input to a resized uint8 grayscale image."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        # gym-super-mario-bros frames are RGB. Use the standard luminance formula.
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(f"Unexpected frame shape {frame.shape}. expected (H, W, 3) or (H, W).")

    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)
